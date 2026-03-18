# SGLang KV Cache 机制与调度执行层分析

## 核心问题
SGLang 作为针对复杂提示词工作流优化的推理框架，其底层是如何管理 KV Cache 的？它的调度层（Scheduler）与模型执行层（Model Executor）又是如何配合来完成高吞吐推理和前缀共享的？

## 关键知识点

### 1. KV Cache 管理机制 (RadixAttention)
SGLang 采用了一套层次化的显存管理机制，核心是基于基数树（Radix Tree）的逻辑映射和底层物理显存池的结合：

*   **逻辑层 - RadixCache (`mem_cache/radix_cache.py`)**：
    *   **职责**：维护一棵全局的 Radix Tree，节点代表一段 Token 序列（Key）及其对应的物理 KV Cache 引用。
    *   **核心逻辑**：当一个新请求到达时，会在 Radix Tree 中执行 `match_prefix`，寻找到最长的公共前缀，从而直接复用之前请求计算好的 KV Cache（这也就是 SGLang 极速处理复杂 Prompt 的核心：**RadixAttention**）。
    *   **生命周期**：请求结束时，如果其包含新的 token 序列，会调用 `cache_finished_req` 将其挂载到树上；如果触发显存不足，会基于 LRU 等策略调用 `evict` 淘汰最久未使用的叶子节点。
*   **分配层 - Allocator (`mem_cache/allocator.py`)**：
    *   **职责**：管理请求到 Token 的映射 (`ReqToTokenPool`) 以及 Token 到物理显存的分配。
    *   **分配策略**：
        *   默认情况下，采用 **Exact Token Allocation (page_size=1)**，即需要几个 Token 就分配几个离散槽位。由 `TokenToKVPoolAllocator` 负责。极大降低显存碎片。
        *   当底层 Attention 算子有特定要求（如 FlashMLA 强制 `page_size=64`），会退化为 **Paged Allocation**。由 `PagedTokenToKVPoolAllocator` 负责。此时行为类似 vLLM 的 PagedAttention。
*   **物理层 - KVPool (`mem_cache/memory_pool.py`)**：
    *   **职责**：持有实际的显存 Tensor，例如 `MHATokenToKVPool`, `MLATokenToKVPool` (针对 DeepSeek 的 MLA 优化) 等。

### 2. RadixTree 核心结构与前缀匹配算法

`RadixCache` 是 SGLang 能够极致复用前缀的核心。基数树（Radix Tree）本质上是一种压缩的前缀树（Trie）。在普通的 Trie 中，每个节点通常只存一个 Token，而在 Radix Tree 中，如果一个序列路径上没有分支，这些 Token 会被压缩合并到一个节点里。

#### 核心数据结构 `TreeNode`
```python
class TreeNode:
    def __init__(self, id: Optional[int] = None, priority: int = 0):
        self.children = defaultdict(TreeNode)  # 子节点，通常用当前子序列的第一个 Token(或 page) 做 key
        self.parent: TreeNode = None           # 父节点指针
        self.key: RadixKey = None              # 这条边代表的 Token IDs 序列
        self.value: Optional[torch.Tensor] = None # 核心：指向物理显存 KVPool 的索引数组 (长度与 key 相同)
        self.lock_ref = 0                      # 锁引用计数（代表当前有多少个 active request 正在使用这个节点）
        self.last_access_time = time.monotonic() # 访问时间，用于 LRU 驱逐策略
```
*   **注意**：`value` 存的不是真实的 KV 数据（浮点数），而是 `ReqToTokenPool` 分配给这串 Token 的**物理槽位索引**（`int64`）。

#### 前缀匹配 (`match_prefix`) 与插入分裂 (`_split_node`)
当一个新请求带着它的输入 `token_ids` 进来时，调度器会调用 `match_prefix`：

1.  **从 Root 开始匹配**：提取请求首个 Token，去 Root 的 `children` 字典里找有没有匹配的子节点。
2.  **边长对比 (`self.key_match_fn`)**：如果找到了分支，就对比请求序列和当前节点的 `node.key` (Token 序列) 在当前方向上能匹配多长。
3.  **精确分割 (Split Node)**：如果一个新请求命中了某个已存在节点的前半段，而在中间产生分叉，`RadixCache` 就会执行 `_split_node`。将被命中的老节点**一分为二**，前一段变成父节点，后一段变成原路径的子节点。而新请求的分支则成为父节点的另一个子节点。

#### 实例演示：插入、分裂与下一轮匹配
假设我们有几个 shared prompt 请求。

**步骤 1：处理请求 A (完全从头计算并建树)**
*   **输入 Tokens**: `[101, 205, 33, 56, 88]`
*   **过程**: 调度器发现树为空，分配 5 个物理槽位（假设分配到 `[0, 1, 2, 3, 4]`）。计算完毕后，调用 `cache_finished_req` 挂载到树上。
*   **树状态**:
    ```text
    Root
     └── (子节点 A) 
         key   : [101, 205, 33, 56, 88]
         value : [  0,   1,  2,  3,  4]  (物理索引)
    ```

**步骤 2：处理请求 B (前缀命中与节点分裂 `_split_node`)**
*   **输入 Tokens**: `[101, 205, 33, 99, 120]`
*   **过程**: 
    1. 首 Token `101` 命中子节点 A。
    2. 对比发现前 3 个 Token `[101, 205, 33]` 是重合的。
    3. 触发 `_split_node`，将节点 A 从中间劈开。调度器拿到物理槽位 `[0, 1, 2]`，只需为后 2 个 Token 申请新槽位（假设分配到 `[5, 6]`）并计算。
*   **树状态 (分叉出现)**:
    ```text
    Root
     └── (中间节点 P) 
         key   : [101, 205, 33]
         value : [  0,   1,  2] 
          │
          ├── (原节点 A 的后半段)
          │   key   : [56, 88]
          │   value : [ 3,  4]
          │
          └── (新节点 B 的后半段)
              key   : [99, 120]
              value : [ 5,   6]
    ```

**步骤 3：处理下一轮请求 C (直接遍历)**
*   **输入 Tokens**: `[101, 205, 33, 99, 120, 500]`
*   **过程**:
    1. 首 token `101` 命中节点 P，吃满其 key `[101, 205, 33]`。
    2. 剩余 `[99, 120, 500]` 继续往下找，首 token `99` 命中节点 B 后半段，再次吃满其 key `[99, 120]`。
    3. 剩余 `[500]` 没有孩子可以匹配，结束。
    4. `match_prefix` 返回合并的物理索引：`[0, 1, 2, 5, 6]`。调度器只算 1 个新 Token `[500]` 即可。

### 3. 调度层与执行层的数据流转与核心数据结构
SGLang 采用了控制平面(Control-Plane)和数据平面(Data-Plane)分离的架构。

**批处理结构的演进：**
`ScheduleBatch` (CPU 调度) -> `ModelWorkerBatch` (跨进程传输) -> `ForwardBatch` (GPU 算子执行)

在这三个结构中，与 KV Cache 强相关的核心字段如下：

1.  **`req_pool_indices` (请求池索引)**
    *   **作用**：表示当前批次里的请求在全局 `ReqToTokenPool` 中占用的逻辑槽位。
    *   **流转**：分配器在分配槽位时，会将这些请求映射到物理内存的关联写到这里。在执行层，通过读取这个索引，可以在 O(1) 的时间内找到这个请求对应的所有历史 Token 的物理存放位置（即 vLLM 中的 block_tables 的概念）。
2.  **`out_cache_loc` (输出缓存位置)**
    *   **作用**：这是一个极其关键的字段。它代表了**调度器提前预分配好的，供当前前向计算存放新生成 KV 数据的物理张量索引**。
    *   **流转**：Scheduler 在组批时向 Allocator 申请得到它。下发到 `ForwardBatch` 后，底层的算子（如 PagedAttention）根本不需要调用内存分配，只需要像执行普通 CUDA 写入操作一样，把结果直接写到 `out_cache_loc` 告诉它的绝对一维/二维地址中去。
3.  **`seq_lens` / `extend_seq_lens` (序列长度)**
    *   **作用**：指示每个请求当前的历史长度和需要新计算的长度。用于注意力掩码（Mask）的生成和寻址偏移。

### 4. 底层 Attention 后端如何消费这些结构
*   **默认 `page_size = 1` (如 FlashInfer 后端)**：
    `req_to_token_pool` 中记录的是完全平铺的一维索引。调度层分配的 `out_cache_loc` 也是完全离散的 N 个 slot。底层的算子通过传入的请求 ID 找到对应的离散历史 Token 地址，并把当前生成的数据写入 `out_cache_loc` 中，完全无视物理上的连续性。
*   **强制 `page_size = 64/128` (如 FlashMLA 后端)**：
    虽然在 `ForwardBatch` 中依然有 `req_pool_indices`，但此时内部的物理池已经被划分为固定大小的页。
    在执行前，例如 `FlashMLABackend.init_forward_metadata()` 中，会执行一个额外的 Triton Kernel `create_flashmla_kv_indices_triton`，它的作用是将一维平铺的逻辑映射，重新打包成针对每个请求的二维 `block_kv_indices`（即经典 PagedAttention 中的 `block_tables`）。底层的硬件优化算子随后使用这个连续成块的索引去 HBM 里高效抓取对齐的整块显存。

### 5. 特殊变体：Mamba Cache 的管理与匹配

Mamba (状态空间模型) 的架构特性决定了它与 Transformer 在缓存管理上的本质区别。Transformer 需要逐 Token 保存 KV 对以供后续 Attention 计算，而 Mamba 的隐藏状态（`conv` 和 `temporal` / `ssm` state）的大小是固定的，与序列长度无关。

由于 Mamba 是循环/状态更新的特性，SGLang 在处理 Mamba 缓存时采用了以下特殊机制：

#### 双重载荷的 `TreeNode`
在 `MambaRadixCache` 中，`TreeNode` 会同时保存两种索引：
*   **`value`**：标准的 KV 槽位或 Token 占位符索引。
*   **`mamba_value`**：这是专门指向 `MambaPool` 中分配的固定大小状态槽位的索引（类型为 `int64`，如果该节点没有存储 Mamba 状态，则为 `None`）。

由于不需要也没有必要在每一个 Token 处都全量保存庞大的 Mamba 状态，缓存通常只存在于按一定长度分块（Chunk）的边界或特定的树节点上。

#### 匹配逻辑（找寻最近有效状态）
在执行 `match_prefix` 时，MambaRadixCache 不仅要看 Token 的重合度，还需要寻找重合路径上**最后一次具有有效 `mamba_value` 的节点**。
*   在树的遍历过程中，只有当遇到 `node.mamba_value is not None` 时，才会更新 `best_last_node`。
*   返回命中结果时，不仅返回物理索引，还会计算出 `mamba_branching_seqlen`（即从这个最长有效节点开始，Mamba 状态需要继续往前推演的位置）。

#### 状态隔离（Copy-on-Write）
因为 Mamba 的状态更新是“破坏性”的（下一个 Token 会直接修改前面的隐藏状态以产生新的状态），为了防止新请求的推演覆盖并污染 Radix 树中被多个分支共享的父节点状态，使用了 **Copy-on-Write (CoW)** 策略：
如果 `cow_mamba` 为真，调度器向 `MambaPool` 申请一段该请求专属的私有空间（通过 `req.mamba_pool_idx` 记录），并将树上命中的共享状态拷贝过来（`copy_from`）。这样该请求后续的模型前向（Forward）更新的只是自己私有的 Mamba 状态。

### 6. 混合架构（Hybrid Mamba & Full Attention）的缓存一致性
对于像 Jamba 这样混合了 Mamba (状态空间模型) 和 Transformer (全注意力) 的架构，其缓存管理的复杂性极高：既需要追踪离散的 Token-level KV Cache，又需要管理固定大小但非线性的 Mamba 隐藏状态。

SGLang 在设计 `MambaRadixCache` 时，明确就是**为了处理这种混合 (Hybrid) 架构而生的**。它是如何保证两者一致性的？主要体现在以下三个层面：

#### 1. 统一的调度前缀树 (`MambaRadixCache`)
SGLang 并没有因为架构混合就去建两棵分开的树，而是将两种缓存状态强行绑定在一棵统一的 `RadixTree` 上。
- **源码位置**：`sglang/srt/mem_cache/mamba_radix_cache.py` -> `class TreeNode`
- **机制**：在 `TreeNode` 中，同时挂载了两个载荷：
  *   **`self.value`**：记录 Transformer 层在该节点对应的历史 Token 的物理显存索引。
  *   **`self.mamba_value`**：记录 Mamba 层在该节点所累积的隐藏状态池索引。
- **意义**：因为 Mamba 和 Attention 共用同一个输入 Token 序列（Prompt），在前缀树的拓扑结构（边的分支、长度）上，它们是绝对同构的。这从源头上保证了“前缀匹配”的物理意义在两种架构下是一致的。

#### 2. 强绑定的生命周期与非对称的锁机制 (Invariant Lock)
虽然挂在同一个节点上，但 Mamba 状态比 KV 状态更稀疏（不是每一个 token 边界都存 Mamba state）。为了保证一致性且避免内存泄露或被误杀，SGLang 设计了一个极具巧思的**非对称双锁机制**：
- **源码位置**：`sglang/srt/mem_cache/mamba_radix_cache.py` -> `TreeNode.__init__` (约 74 行)
    ```python
    # invariant: 如果 mamba_lock_ref 被锁定，full_lock_ref 必须被锁定；
    # 反之如果 full_lock_ref 锁定，mamba_lock_ref 不一定锁定。
    self.full_lock_ref = 0    # 控制 Attention KV 的锁
    self.mamba_lock_ref = 0   # 控制 Mamba 状态的锁
    ```
- **代码体现**：在 `inc_lock_ref` (加锁) 和 `dec_lock_ref` (解锁) 时（约 790 行）：
    ```python
    # 只要有 Mamba 状态被保护，就同时增加 full_lock_ref 的保护
    if node.mamba_value is not None:
        self.mamba_evictable_size_ -= len(node.mamba_value)
        self.mamba_protected_size_ += len(node.mamba_value)
    ```
- **意义**：这保证了：只要有请求正在依赖某个 Mamba 状态进行推理，支撑这个状态一路走过来的所有 Transformer 的 KV Cache 绝对不会被系统驱逐（Evict），避免了“状态还在，KV 丢了”的灾难性不一致。

#### 3. LRU 双链表的降级驱逐策略
当 GPU 显存吃紧，触发 LRU 驱逐时，怎么保证两者释放的一致性？
- **源码位置**：`sglang/srt/mem_cache/mamba_radix_cache.py` 
- **机制**：`MambaRadixCache` 内部维护了**两条平行的双向链表**：`self.full_lru_list` 和 `self.mamba_lru_list`。
- **降级释放流程**：
    1. **只释放 Mamba 状态 (`evict_mamba`, 约 728 行)**：系统会安全地把一些叶子节点的 `node.mamba_value` 置为 `None`，并回收 Mamba 池内存。但节点本身和它对应的 Attention KV Cache `node.value` **依然存活**。因为以后有新请求来，最多就是多推演几步 Mamba 状态，但至少不用重新算 KV。
    2. **彻底销毁节点 (`evict_full`, 约 763 行)**：只有当普通 KV Cache 都不足时，才会彻底把叶子节点从树上摘除。如果它身上还有 `mamba_value`，则顺带一起释放。

#### 4. 混合调度总结流程
当一个新的混合模型请求进入：
1. 调度器拿 Prompt 去问 `MambaRadixCache`。
2. 树返回一个统筹的 `MatchResult`，里面包含两个截然不同的指令：
   * `device_indices`：拿去给 Attention 层，告诉它“直接拿这些物理槽位作为过去的历史 KV”。
   * `mamba_branching_seqlen` / 复制好的私有 `mamba_pool_idx`：拿去给 Mamba 层，告诉它“从序列的第 X 个位置开始推算新的状态”。
3. 组装好的统一 `ForwardBatch` 被传递给执行层。

#### 5. 执行层中的算子级组装 (以 Nemotron-H/Jamba 为例)
在混合模型的执行层 (例如 `sglang/srt/models/nemotron_h.py`)，**同一份** `ForwardBatch` 会像流水线一样依次经过两类不同的计算层，两者分别各取所需：

*   **Transformer 注意力层 (`AttentionDecoderLayer`)**:
    *   **核心算子**：直接使用 SGLang 标准的 `RadixAttention` 模块（内部根据环境包装了 `FlashInfer` / `Triton` 等）。
    *   **输入组装**：调用 `self.attn.forward(q, k, v, forward_batch)`。`RadixAttention` 会从 `forward_batch` 中提取刚才树匹配返回的 `req_pool_indices` 和 `out_cache_loc`，只去内存池中读取或写入局部的 KV 数据，就像普通的 Transformer 一样。
*   **Mamba 状态层 (`MambaDecoderLayer`)**:
    *   **核心算子**：使用针对 Mamba 优化的 `MambaMixer2` 以及对应的 `Mamba2AttnBackend`。
    *   **输入组装**：调用 `attn_backend.linear_attn_backend.forward(..., forward_batch=forward_batch)`。此时，算子根本不会看传统的 `out_cache_loc`，而是会去读取 `forward_batch.mamba_pool_idx` 以及推演的起点 `mamba_branching_seqlen`，执行状态的更新。

这样，同一个请求的两种截然不同的上下文缓存机制被完美地融合在了同一个前向传播批次（Batch）中。

#### 6. 实例演示：混合架构下的请求匹配与 CoW 机制
假设我们部署了一个 Jamba 模型，并设置 Mamba 状态每隔 100 个 Token 缓存一次（Chunk size = 100）。

**步骤 1：处理初始系统提示词 (Prompt A)**
*   **输入 Tokens**: 长达 150 个 Token 的系统提示词 `[T1, T2, ..., T150]`。
*   **过程**: 
    1. 首发请求，树为空。执行计算。
    2. 计算完成后挂树。因为 Chunk size 为 100，所以在 Token 100 的边界处，节点会**拥有 Mamba 状态**，而 Token 150 的叶子节点也有一份完整的 Mamba 状态。
*   **树状态**:
    ```text
    Root
     └── (中间节点 Chunk1)
         key         : [T1 ... T100]
         value       : [0 ... 99] (物理 KV 槽位)
         mamba_value : [M_Idx_1]  (池中的状态张量索引)
          │
          └── (叶子节点 A)
              key         : [T101 ... T150]
              value       : [100 ... 149]
              mamba_value : [M_Idx_2]
    ```

**步骤 2：多并发分支接入 (Prompt B 与 Prompt C)**
*   **输入**: 两个用户并发访问。
    - **请求 B**: 系统提示词 + `"Query B"` (150 + 20 tokens)
    - **请求 C**: 系统提示词 + `"Query C"` (150 + 30 tokens)
*   **调度器匹配 (`match_prefix`)**:
    1. 两个请求都完全命中了叶子节点 A，重合度为 150 个 Token。
    2. 发现叶子节点 A 的 `mamba_value` 不为 None (`[M_Idx_2]`)。
*   **触发 Copy-on-Write (写时复制)**:
    - Mamba 状态不能被共享覆盖。此时调度器向 `MambaPool` 分别为 B 和 C 申请了新的私有状态槽位：`M_Idx_B` 和 `M_Idx_C`。
    - 执行 `copy_from`：将 `M_Idx_2` 的真实状态数据深拷贝到 `M_Idx_B` 和 `M_Idx_C` 中。
*   **下发计算**:
    - **对 Attention 层**：调度器告诉它们：“你们的历史 KV 就是物理槽位 `[0 ... 149]`，直接读”。
    - **对 Mamba 层**：调度器告诉请求 B：“你的历史状态在 `M_Idx_B`，拿着它去算最后的 20 个 Token”。告诉请求 C：“拿着 `M_Idx_C` 去算最后的 30 个 Token”。
*   **隔离保护**：请求 B 和 C 计算时，只会弄脏各自的 `M_Idx_B` 和 `M_Idx_C`，最初的公共状态 `M_Idx_2` 完好无损地躺在树里，随时准备迎接下一个并发用户的匹配。

### 7. Mamba 算子的底层续算逻辑 (State Continuation)
当 `MambaMixer2` 算子拿到调度器给的 `mamba_pool_idx` 和后续的新 Token 时，它是如何“在之前的基础上继续算”的？

在源码 `sglang/srt/layers/attention/mamba/mamba.py` 中的 `MambaMixer2.forward` 方法里，核心分为两个状态的前向传播：

1. **一维卷积状态 (`conv_state`) 续算**：
   * Mamba 第一步是一个 1D 卷积，它的感受野通常是前 $K$ 个 Token。
   * 算子会从 `MambaPool` 中把对应的 `conv_state` 取出来。此时里面正躺着之前树匹配回来的（比如系统提示词最后几个 Token 的）隐藏状态。
   * 使用自定义的 `causal_conv1d_fn_triton` 内核，将之前存在的 `conv_state` 作为 `initial_states` 送入卷积。算出来的结果同时会**覆盖原有的 `conv_state` 张量**，作为下一步计算的历史基底。

2. **核心 SSM 状态 (`temporal/ssm_state`) 续算**：
   * 算子将从池中取出 `ssm_state`，这是 Mamba 递归推演的核心记忆矩阵。
   * 算子调用 `mamba_chunk_scan_combined`，把上一步保存的 `ssm_state` 传进去。这就相当于在数学上把 $h_t = A \cdot h_{t-1} + B \cdot x_t$ 里的 $h_{t-1}$ 从历史快照中直接接上了。
   * 同样地，新的 $h_t$ 算出来后，内核会把池子里的状态也更新掉。

因为调度器提前做好了 **CoW (写时复制)** 隔离，算子在这里只管“读取历史 -> 算入新 Token -> 直接覆盖旧历史”，完全不需要担心会破坏别的前缀状态。这就是 Mamba 层如此丝滑高效的原因。

### 8. 进阶：当 Attention KV 和 Mamba 状态“命中长度不一致”时的完美降级
这是混合架构中最棘手的一个 Edge Case（边界情况）：由于 Mamba 状态是稀疏保存的（比如每 100 个 Token 存一次），而 Attention KV 是稠密保存的（每个 Token 都有）。
如果在前缀匹配时，一个新请求（比如总长 150）在树上命中了 130 个 Token 的前缀，就会出现**“潜在错位”**：
*   **Attention 理论命中**：130 个 Token。
*   **Mamba 实际命中**：最近的有效状态只能回退到第 100 个 Token。

如果是错位的，给模型的 `input_ids` 到底该是 20 个（Attention 想要的）还是 50 个（Mamba 想要的）？如果只传 20 个，Mamba 缺少中间 30 个 Token 的 `hidden_states` 是无法推演的！

#### SGLang 的破局之道：木桶效应与截断降级 (Truncation)
SGLang 并没有选择去搞复杂且容易出错的“缓存 hidden_states 并在底层算子里拼接”的黑科技，而是采取了一种极其简单、优雅且鲁棒的策略：**向短板看齐 (降级匹配)**。

在 `sglang/srt/mem_cache/mamba_radix_cache.py` 的 `_match_post_processor` 源码中：
```python
# value 是原本可以命中 130 个的物理索引列表
# best_value_len 是最近一个有 mamba_value 的长度（即 100）

# 强行截断！
value = value[:best_value_len] 
if value:
    value = torch.cat(value)
```
**原理解析**：
1. **强制截断**：`match_prefix` 发现 Mamba 只能命中 100，它会**直接把 Attention 的命中结果也强行截断到 100**。
2. **统一调度**：最终调度层拿到的 `prefix_lens = 100`。于是调度层会给该请求下发 `extend_lens = 50` 的指令。
3. **完美对齐**：对于 Transformer 层和 Mamba 层来说，现在世界线统一了。它们收到的都是同一个 `input_ids`（长度 50，对应第 101~150 个 Token）。Transformer 拿着前 100 个历史 KV 算这 50 个新 KV；Mamba 拿着第 100 个位置的初始状态推演这 50 个新状态。
4. **延迟补偿 (`mamba_branching_seqlen`)**：既然退回到了 100 重新算 101~150，为了以后别的请求不再吃亏，匹配结果会附带一个 `mamba_branching_seqlen = 100` 或 `100 的倍数`。这其实是一个**追踪指令 (Tracking Instruction)**！它告诉 Mamba 算子：“在你这次前向计算跑这 50 个 Token 的时候，顺便在符合 Chunk 边界的地方打个快照存到树里！”

### 总结
这就是极致系统工程的魅力：**大道至简**。
SGLang 并没有去保存庞大的中间层 `hidden_states`，也没有写复杂的底层特征拼接 Kernel。面对异构缓存的错位，它通过在匹配结果返回前做一次极简的**木桶短板截断**，强行保证了 `Input IDs` 和 `Hidden States` 的形状在宏观和微观上永远对齐。

## 目录与文件溯源
- **RadixCache 实现**：`sglang/srt/mem_cache/radix_cache.py`, `sglang/srt/mem_cache/mamba_radix_cache.py` (Hybrid/Mamba 变体)
- **内存池与 Allocator**：`sglang/srt/mem_cache/allocator.py`, `sglang/srt/mem_cache/memory_pool.py` (`MambaPool`, `HybridReqToTokenPool`)
- **数据结构定义**：`sglang/srt/managers/schedule_batch.py` (`ScheduleBatch`, `ModelWorkerBatch`), `sglang/srt/model_executor/forward_batch_info.py` (`ForwardBatch`)
- **调度显存分配逻辑**：`sglang/srt/mem_cache/common.py` (`alloc_for_extend`, `alloc_for_decode`)
- **执行层算子转换**：`sglang/srt/layers/attention/flashmla_backend.py` (`init_forward_metadata` 对 `block_kv_indices` 的重新打包)
