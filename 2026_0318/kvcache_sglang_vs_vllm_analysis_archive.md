# 会话归档：SGLang 与 vLLM v1 的 KV Cache 及异构架构管理深度分析

## 核心问题
本次对话深入探讨了当前主流推理框架 SGLang 和 vLLM v1 是如何管理 KV Cache 的，特别是它们在处理 **Mamba（状态空间模型）** 与 **混合架构（Hybrid Mamba & Full Attention）** 时的底层设计哲学、数据结构流转以及显存分配策略。

## 关键知识点

### 1. SGLang 的 KV Cache 管理 (RadixAttention)
*   **设计哲学**：追求极致的前缀复用与极细粒度的显存控制。
*   **核心实现**：
    *   **基数树 (`RadixCache`)**：使用具有父子拓扑结构的 `TreeNode` 记录 Token 前缀。
    *   **细粒度分配**：默认 `page_size=1`，物理显存分配做到了零内部碎片。通过 `out_cache_loc` 直接将绝对物理地址下发给底层的算子执行。
*   **Mamba 异构处理**：
    *   **双重载荷节点**：`TreeNode` 同时保存 Transformer 的 `value` 和 Mamba 的 `mamba_value`，辅以非对称的双锁机制（`mamba_lock` 必然触发 `full_lock`）保证一致性。
    *   **写时复制 (CoW)**：由于 Mamba 是原地的递归推演，为了保护树上的公共前缀不被污染，多分支并发命中时会分配私有槽位，使用 `copy_from` 进行状态快照克隆。
    *   **错位降级（木桶截断）**：当 Attention 命中长度长于 Mamba 命中长度时，强行截断向 Mamba 的短板对齐，同时打上 `mamba_branching_seqlen` 追踪快照标记。

### 2. vLLM v1 的 KV Cache 管理 (状态机与哈希)
*   **设计哲学**：追求极致的分布式调度性能与最低的框架/算子通信开销。
*   **核心实现**：
    *   **哈希前缀缓存**：摒弃复杂的树结构，使用扁平的 `BlockHashToBlockMap`，以 `block_size=16` 为粒度计算 Hash 判断命中。
    *   **状态机同步**：调度器（`SchedulerOutput`）不发送庞大的物理块地址（`block_tables`），只发送极简指令，依赖底层的 Worker 根据一致的状态机在本地重演和重建 `block_tables`。
*   **Mamba 异构处理**：
    *   **解耦式门面模式**：通过 `HybridKVCacheCoordinator` 聚合 `FullAttentionManager` 和 `MambaManager`。
    *   **不动点迭代截断**：`find_longest_cache_hit` 在遇到错位时，让 Attention 从左向右扫，Mamba 从右向左退，然后强制对齐，直到不再改变。
    *   **Mamba 专用“大块”与“阅后即焚”**：vLLM 不建立独立的 `MambaPool`，而是调整 Mamba Group 的 `page_size_bytes` 来塞入整个系统状态。在 `"align"` 模式下，推演新 Token 时申请 1 个新 Block，算完立即 `free_blocks` 释放旧 Block，极限压榨显存。

## 代码与架构溯源
### SGLang 关键溯源
*   **树与 Mamba 节点**：`sglang/srt/mem_cache/mamba_radix_cache.py` (重点关注 `_match_post_processor` 里的 CoW 逻辑，以及 `TreeNode` 的双锁定义)。
*   **调度下发指令**：`managers/schedule_batch.py` (关注 `out_cache_loc` 的生成)。

### vLLM v1 关键溯源
*   **物理池与哈希表**：`vllm/v1/core/block_pool.py`
*   **策略层大师**：`vllm/v1/core/single_type_kv_cache_manager.py` (特别是 `MambaManager.allocate_new_blocks` 里的滚动回收逻辑 `free_blocks`)。
*   **不动点迭代截断算法**：`vllm/v1/core/kv_cache_coordinator.py` (`find_longest_cache_hit`)。

## TODOs 与 下一步计划
- [ ] **按阅读指南实战**：按照 `2026_0318/kvcache_reading_guide.md` 给出的路线，进入对应的项目目录通读一遍源码实现。
- [ ] **关注算子层**：深入底层的 `FlashMLA` / `FlashInfer` / `MambaMixer2` 的 Triton 或 CUDA Kernel，查看它们到底是怎么消费传入的物理显存地址的。
- [ ] **监控指标对比**：在实际机器上跑一下两者，观察开启 Mamba 混合架构后，吞吐量和 GPU 显存利用率（碎片率）的具体差异。
