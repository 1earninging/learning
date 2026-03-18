# vLLM V1 架构的 KV Cache 管理机制分析

## 核心问题
在 vLLM 的 `v1` 版本架构中，KV Cache 是如何进行精细化管理的？显存分配的信息是如何在调度层（Scheduler）、模型执行层（Model Executor）与底层注意力算子（Attention）之间流转的？

## 关键知识点

### 1. 层次化的 KV Cache 管理类设计
vLLM v1 的 KV Cache 管理抽象非常清晰，采用了一种组合与策略分发的模式：

*   **`BlockPool` (`vllm/v1/core/block_pool.py`)**: 
    *   **职责**：最底层的物理块管理器。它维护了可用空闲块（Free blocks）的队列以及基于哈希（Block Hash）的前缀缓存池（`BlockHashToBlockMap`）。
    *   **分配粒度**：它是严格按 `block_size` (比如 16 个 token) 来分配和回收物理槽位的。
*   **`SingleTypeKVCacheManager` (`vllm/v1/core/single_type_kv_cache_manager.py`)**:
    *   **职责**：负责管理**单一类型**注意力机制的缓存分配算法。
    *   **多态实现**：vLLM v1 对不同的架构设计了不同的继承子类，如：
        *   `FullAttentionManager`: 处理传统的大模型，所有历史块都要保留。
        *   `SlidingWindowManager`: 处理滑动窗口注意力（如 Mistral），会自动回收滑出窗口的历史块。
        *   `MambaManager`: 处理 Mamba 这类 SSM 模型。
        *   `CrossAttentionManager`: 用于 Encoder-Decoder 架构。
*   **`KVCacheManager` (`vllm/v1/core/kv_cache_manager.py`)**:
    *   **职责**：这是一个**聚合门面（Facade）**。因为一个模型可能同时存在多种注意力机制（例如混合架构），它内部维护了一个 `managers: list[SingleTypeKVCacheManager]`。
    *   外部 Scheduler 只和它交互，调用 `allocate_slots()`。它会迭代所有的子 manager，拿到各自的物理 `block_ids`，然后打包成 `KVCacheBlocks` 对象返回。

### 2. 调度层的显存分配：如何组装并传给模型？

与旧版本或 SGLang 不同，vLLM v1 在**调度层 (Scheduler) 和 模型执行器 (Worker) 之间解耦了显存映射数据**。

#### A. 调度器分配 (Scheduler)
在 `Scheduler.schedule()` 中：
1. 取出 Request。
2. 调度器调用 `kv_cache_manager.allocate_slots()` 尝试为其分配新的 Block。
3. 调度器的产出物是一个极度精简的 `SchedulerOutput` 对象 (`vllm/v1/core/sched/output.py`)。

**注意关键点**：`SchedulerOutput` 中**并没有直接携带**每个请求对应的物理 `block_table` 或 `block_ids`！它只传输了 `num_scheduled_tokens` (要算多少 token) 和请求的 ID 等元数据。

#### B. 模型层重构 Block Tables (Model Executor)
为什么 `SchedulerOutput` 不传具体的显存映射？因为在 vLLM v1 的架构中，每个 Worker（GPU 进程）内部**自己维护了一份等价的局部 `KVCacheManager` / `BlockPool`**。
*   只要 Scheduler 决定调度某个请求，各个 Worker 内部的局部状态机收到指令后，会同步进行同样的逻辑推演（甚至由调度器通过 RPC 只传轻量级的增量数据），在 Worker 端**重构出真正的 `block_tables`**。

#### C. Attention 算子如何消费？
当指令落到底层的 `ModelRunner` 和具体的模型时：
*   在 Attention 层前向传播前，会通过内部收集好的元数据，生成一个二维的 `block_tables` 张量：形状为 `[num_seqs, max_num_blocks_per_seq]`。
*   底层的 PagedAttention 算子拿到：
    1.  `Q, K, V` 向量。
    2.  `block_tables` (记录每个请求的物理页块 ID)。
    3.  `slot_mapping` 或 `out_cache_loc` (记录新生成的 K/V 需要写入的一维物理绝对地址)。
*   算子利用 `block_tables` 去读取分散的历史缓存，同时将计算完的新 K/V 利用 `slot_mapping` 写回物理池。

### 3. 一个请求进来的逐层分配生命周期 (Lifecycle)
以一个全新的 Prompt 请求（比如 100 个 Token）进入混合模型（如包含 Full Attention 和 Mamba）为例，整个过程如下：

1.  **调度器入口 (`Scheduler.schedule`)**：
    *   调度器在构建 Batch 时，发现这是一个新请求，计算出它需要分配 $\lceil 100 / 16 \rceil = 7$ 个 Block。
    *   调用 `self.kv_cache_manager.allocate_slots(...)`。
2.  **聚合分配器分发 (`KVCacheManager.allocate_slots`)**：
    *   `KVCacheManager` 检查全局剩余块是否大于 7。
    *   通过 `self.coordinator.allocate_new_blocks`，向底下的每一个 `SingleTypeKVCacheManager` 下发指令。
3.  **单类型管理器层各自认领 (`SingleTypeKVCacheManager.allocate_new_blocks`)**：
    *   由于是混合模型，这里会有多个管理器并行工作。
    *   **对于 `FullAttentionManager`**：它从 `BlockPool` 中真实地划拨 7 个空闲块（例如 ID 为 `[1, 2, 3, 4, 5, 6, 7]`），追加到该请求在其内部的追踪列表中。
    *   **对于 `MambaManager`**：它根据其特有的分配逻辑（通常不需要像 Attention 那样为每个 Token 保存），向自己管理的池中申请对应的状态块（例如分配了块 `[8]`），追加到追踪列表。
4.  **返回分配结果并下发指令**：
    *   各个管理器分配的 Block ID 会汇聚成 `KVCacheBlocks` 对象返回给调度层。
    *   然而，调度器组装给底层的 `SchedulerOutput` 时，**丢弃了这些具体的物理 Block ID**，只告诉底层：“请求 A，新算 100 个 Token”。
5.  **底层 Worker 同步状态机重演**：
    *   底层 GPU 进程收到请求 A 要算 100 个 Token 的指令。
    *   它立刻唤醒自己进程内的 `KVCacheManager`。因为调度策略是确定性的，GPU 端管理器同样向 GPU 端的 `BlockPool` 申请了 7 个块（此时因为进程同步，GPU 端 `BlockPool` 弹出的空闲块必定也是 `[1...7]` 和 `[8]`）。
    *   在 GPU 本地，这 7 个块的 ID 被组装进了即将传入 CUDA Kernel 的 `block_tables` 张量中！

### 4. 混合模型中的前缀匹配与“降级错位”处理 (HybridKVCacheCoordinator)
前面我们讲到，在一个混合模型（如 Jamba：Full Attention + Mamba）中，由于它们分属不同的 Group，调度器是如何统筹它们的前缀命中结果的？如果一个命中了 100，一个命中了 150 怎么办？

在 `vllm/v1/core/kv_cache_coordinator.py` 的 `HybridKVCacheCoordinator` 类中，有一个极其核心的函数 `find_longest_cache_hit`。它完美解决并统一了这个问题，它使用了一种**“不断向短板妥协”的不动点迭代算法 (Iterative Fixed-point Algorithm)**。

#### 匹配算法流程：
当一个请求拿着一串 Token (比如算出了 10 个 block 的 Hash 值) 进来时：

1. **先问 Full Attention (从左到右扫)**：
   调度器规定 `FullAttentionManager` 总是第一个发话。
   它从前往后查 Hash，发现前 8 个 Block 都命中了（长度 8 * 16 = 128）。此时，`curr_hit_length` 被更新为 128。

2. **再问 Mamba (或者 Sliding Window) (从右向左扫)**：
   轮到 `MambaManager` 发话了。调度器对它说：“Attention 最多只能命中 128，你只能在这个范围内找。”
   `MambaManager` 从 128 开始**从右向左**往回找，发现 128 处没有有效对齐的状态，退到了第 100 个 Token 处才找到了有效的快照 Block。
   此时，`curr_hit_length` 被残忍地削减（Truncate）到了 100。

3. **重新循环 (Fixed-point iteration)**：
   因为长度被削减了，调度器会再循环一次。
   回到 `FullAttentionManager`，由于长度变成了 100，Attention 会乖乖地把刚才命中的后 28 个 Token 的缓存索引全部丢弃（`del blks[num_blocks:]`），强制和 Mamba 保持一样的长度 100！

4. **跳出条件**：
   如果在一轮循环中，没有任何一个 Manager 继续缩减这个长度，说明大家达到了一个统一的“最大公约数”木桶底。循环打破，返回 `100` 给模型执行层。

#### 结论
这也就是为什么在混合模型下，虽然不同的 Manager 管理着自己独立的逻辑流，但最终下发给模型计算时，**所有层的 `input_ids` 永远是等长且对齐的**。
**vLLM 选择了在调度器层面做强行截断拦截**，以此换来底层算子的纯粹与简单。

## 与 SGLang 的对比总结
*   **分配粒度**：SGLang 默认 `page_size=1`（按 Token 精确分配）；vLLM 坚持使用大块（默认 `block_size=16`），如果用不满就会产生内部显存碎片。
*   **前缀树实现**：SGLang 有一棵全局的、结构化的 Radix Tree；而 vLLM 的 Prefix Caching 是一种扁平化的哈希表 (`BlockHashToBlockMap`)，通过将一串 Token 序列 Hash 化成字符串来比对是否命中。
*   **架构抽象**：vLLM v1 通过 `SingleTypeKVCacheManager` 把逻辑解耦得非常干净，对 Sliding Window、Cross Attention 这种非标准的 Attention 适配性极强。