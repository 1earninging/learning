# MindIE-LLM 混合注意力 (Mamba + Full Attention) 架构演进设计方案

结合业界最先进的推理框架（SGLang 和 vLLM v1）的演进经验，以及 MindIE-LLM 目前的 C++ 架构基座（如 `src/block_manager` 和 `src/scheduler`），如果 MindIE 需要支持类似 Jamba 这样的异构混合模型，建议采用**“统一抽象，分离控制，底层拼接”**的设计哲学。

以下是针对 MindIE-LLM 源码结构的设计建议：

## 一、 显存管理层重构 (Block Manager 改造)
当前 MindIE 只有针对 Transformer 的 `SelfAttnBlockManager`（继承自 `BlockSpaceManager`），这是一种面向 Token/Block 的管理方式。而 Mamba 需要的是固定大小的“状态块（State Slot）”。

### 1. 抽象多态的 Block Manager
在 `src/block_manager` 下引入类似于 vLLM v1 的组合模式：
*   **新建 `MambaStateBlockManager`**：专门用于管理 Mamba 层的 `conv_state` 和 `ssm_state` 张量池。它的分配粒度不再是 16 个 Token 的 Block，而是每个请求分配 1 个固定大小的状态槽（State Index）。
*   **新建 `HybridBlockCoordinator` (门面协调器)**：它继承自 `BlockSpaceManager`，并在内部持有 `SelfAttnBlockManager` 和 `MambaStateBlockManager` 的指针。外部的 `Scheduler` 不感知混合架构，只需要调用 `HybridBlockCoordinator->Allocate(seqGroup)`，协调器会在内部同时向两个 Manager 申请各自的显存并组装返回。

### 2. Mamba 状态的生命周期 (利用现有的 CoW 机制)
由于 Mamba 状态是不断原地覆盖更新的，在面对多分支并发（如 Beam Search, Speculative Decoding）时会有状态污染问题。
*   MindIE 已经在 `src/block_manager` 中拥有了 `copy_on_write_tracker.cpp` 和 `ref_counter.cpp`。
*   **建议策略**：利用 MindIE 现成的 CoW 机制，当一个 `Sequence` 发生 Fork 时，不仅增加 KV Block 的引用计数，也增加 Mamba State Index 的引用计数。当子序列需要真正执行 Mamba 推演写入新状态时，触发 **Copy-on-Write**，从 NPU 的 `MambaPool` 中申请一个新槽位，将老状态克隆过去再进行前向传播，从而完美保护共享前缀的状态。

## 二、 调度器与前缀缓存对齐 (Scheduler & Prefix Cache)
由于 `SelfAttn` 是每个 Token 都存 KV，而 `Mamba` 为了节省显存可能每 64 或 128 个 Token（Chunk 边界）才保存一次状态快照，这就导致了在 `PrefixCacheBlockAllocator` 匹配历史前缀时，两者的命中长度不一致。

### 1. 不动点截断对齐算法 (Iterative Truncation)
在 `src/scheduler` 在构建 `ScheduleBatch` 前的匹配阶段，必须引入“木桶效应短板对齐”机制：
*   先让 `SelfAttnBlockManager` 查 RadixTree / Hash，可能命中了 130 个 Token。
*   再让 `MambaStateBlockManager` 查，发现最近的有效状态快照只有第 100 个 Token 的。
*   **强制截断**：调度器强制将 `SelfAttn` 的匹配结果砍回 100。
*   **统一下发**：调度器认为该请求只命中了 100，给底层下发 `extend_lens = 50` (比如原长 150)。

### 2. 追踪快照标记 (`mamba_branching_seqlen`)
在 `Sequence` 或 `Request` 的结构体中，增加一个字段 `mamba_track_seqlen`。当调度器强行退回 100 时，把 100 塞进这个字段。这等于告诉底层的 NPU 算子：“在你跑剩下的 50 个 Token 的途中，记得在走到第 100 的边界时，顺便往显存池里打个快照存下来！”

## 三、 C++ 到底层 Python/算子的结构体传递
在 `src/server` 与 `mindie_llm/runtime` 通信的接口中（如 `ForwardBatch` 或张量列表中），需要同时下发两套并行的索引：
1.  **`block_tables`**：供 Attention 层使用，表示历史 KV 存放在哪。
2.  **`mamba_state_indices`**：供 Mamba 层使用，表示历史 SSM 状态存放在 `MambaPool` 的哪个槽位。

## 四、 NPU 算子与模型执行层 (Runtime & Kernels)
在 `mindie_llm/modeling` 对应的具体模型文件（如 Jamba 的封装实现中），不同层各取所需：
*   整个模型接收的 `input_ids` 长度是被调度器截断对齐后的长度（比如上面例子的 50）。
*   进入 Attention 层时，使用传统的 PagedAttention 算子读取 `block_tables`。
*   进入 Mamba 层时：
    *   **历史状态回溯**：NPU 内核内部，通过 `mamba_state_indices` 去显存里提取上次保存在第 100 个 Token 处的 `ssm_state` 和 `conv_state`。
    *   **NPU 底层缝合**：将提取出的状态作为初始状态（`initial_states`），喂给昇腾 NPU 优化的 Causal Conv1d 和 Mamba Chunk Scan 融合算子。
    *   **输出对齐**：算子计算出 50 个新 Token 的状态后，将最新的状态**原地写回 (In-place update)** 或写到 CoW 分配的新槽位中。而返回给 PyTorch/MindSpore 框架的 `Hidden_States` 依然是规整的 `[50, hidden_size]`，保证了残差连接的正常相加。

## 总结：MindIE 演进优势
1.  **复用基建**：无需抛弃现有的 `SelfAttnBlockManager`，而是通过装饰器/协调器模式无痛接入 `MambaManager`。
2.  **安全高效**：利用 MindIE 成熟的 `copy_on_write_tracker` 保护 Mamba 并发状态。
3.  **算子纯粹性**：通过在 C++ `Scheduler` 层强制做前缀长度截断对齐，使得昇腾底层算子的实现变得简单，不用在 NPU 上做复杂的特征拼接与切片。
