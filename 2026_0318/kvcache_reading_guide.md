# SGLang & vLLM v1 KV Cache 源码阅读指南

为了透彻理解这两个顶尖推理框架在 KV Cache 管理上的设计差异，建议按照**“从宏观调度到微观结构，再到底层算子消费”**的顺序进行阅读。

---

## Part 1: SGLang KV Cache 源码阅读顺序

SGLang 的核心灵魂是 **RadixTree (基数树)** 以及对 Mamba 的特殊处理。它的代码在 `sglang/srt/` 目录下。

### 1. 基数树与逻辑映射层 (核心骨架)
这是 SGLang 最具特色的地方，建议最先阅读：
*   **`mem_cache/radix_cache.py`**
    *   看 `class TreeNode`: 理解 `key` (Token)、`value` (物理索引)、`lock_ref` (锁) 的作用。
    *   看 `match_prefix()` 和 `_match_prefix_helper()`: 理解请求进来是如何顺着树往下找相同前缀的。
    *   看 `_split_node()`: 这是动态树结构的神来之笔，理解当匹配发生错位时，如何把一个节点劈成两半。
*   **`mem_cache/mamba_radix_cache.py`** (进阶：Mamba 变体)
    *   看 `class TreeNode` 里的变异：新增了 `mamba_value` 和非对称的 `mamba_lock_ref`。
    *   看 `_match_post_processor()`: 理解 Mamba 的 CoW (Copy-on-Write) 机制是如何被触发的，以及如何申请私有槽位。

### 2. 物理内存池 (肌肉)
看看上面树节点里的 `value` 到底指向了物理显存的什么地方：
*   **`mem_cache/memory_pool.py`**
    *   看 `class ReqToTokenPool`: 理解 Request ID 是如何映射到一堆一维离散 Token 槽位的。
    *   看 `class MambaPool`: 看看 Mamba 状态快照是怎么存的（固定大小的离散槽位）。
*   **`mem_cache/allocator.py`**
    *   看 `TokenToKVPoolAllocator`: 了解 `page_size=1` 的精细分配。
    *   看 `PagedTokenToKVPoolAllocator`: 了解为了配合 FlashMLA 强制降级到 `page_size=64` 时的分配。

### 3. 调度层与执行层 (指挥官)
看看系统是如何把树和内存池串联起来的：
*   **`managers/schedule_batch.py`**
    *   看 `class ScheduleBatch` 及其 `get_model_worker_batch()`: 理解统一下发的批次结构。重点看 `req_pool_indices` 和 `out_cache_loc`。
*   **`mem_cache/common.py`**
    *   看 `alloc_for_extend` 和 `alloc_for_decode`: 理解调度器是如何拿着请求去申请物理槽位并装填 `out_cache_loc` 的。

### 4. 算子执行层 (消费者)
*   **`layers/attention/flashmla_backend.py`** (可选，看异构处理)
    *   看 `init_forward_metadata()`: 重点看那个 Triton Kernel `create_flashmla_kv_indices_triton`，看它如何把一维的离散槽位，重新打包成底层的二维 `block_tables`。
*   **`models/nemotron_h.py`** (混合模型组装)
    *   看 `AttentionDecoderLayer.forward` 和 `MambaDecoderLayer.forward`，看它们如何从同一个 `forward_batch` 里各取所需。

---

## Part 2: vLLM v1 KV Cache 源码阅读顺序

vLLM v1 的核心在于 **状态机架构、哈希匹配与模块化解耦**。它的代码集中在 `vllm/v1/core/` 目录下。

### 1. 物理池与数据结构 (基础)
首先要理解 vLLM 里一切皆 "Block" 的概念：
*   **`core/block_pool.py`**
    *   看 `class BlockPool`: 它是物理块的唯一持有者。看它的 `get_new_blocks` 和 `free_blocks`，理解极其轻量的整数 ID 弹出和塞回（Free Queue）。
    *   看 `class BlockHashToBlockMap`: 理解前缀缓存是如何用字典（Hash Map）而不是树来实现的。
*   **`kv_cache_utils.py`**
    *   看 `hash_block_tokens()`: 理解前缀复用的触发条件（算哈希）。

### 2. 策略分发层 (核心大脑)
这是 vLLM v1 最优雅的设计：
*   **`core/single_type_kv_cache_manager.py`**
    *   先看基类 `SingleTypeKVCacheManager`，了解有哪些通用方法。
    *   重点看 `FullAttentionManager`: 它是经典 Transformer，看它的 `allocate_new_blocks` 是如何按需分配新 Block 的。
    *   **超级重点看 `MambaManager`**: 
        *   看 `find_longest_cache_hit`: 理解它为什么要求强制 `alignment` 对齐。
        *   看 `allocate_new_blocks` 和 `remove_skipped_blocks`: 深刻体会“只分配 1 个新块”和“阅后即焚（Free 旧块）”的循环滚动逻辑！

### 3. 门面协调器 (包工头)
看看上面的多个 Manager 是怎么被统筹的：
*   **`core/kv_cache_coordinator.py`**
    *   重点看 `class HybridKVCacheCoordinator` 里的 `find_longest_cache_hit()`。
    *   欣赏里面那个著名的 **“不动点迭代 (Fixed-point iteration)”**：看 Attention 和 Mamba 是如何互相压价，最后强制向短板截断对齐的。
*   **`core/kv_cache_manager.py`**
    *   看 `KVCacheManager.allocate_slots()`: 这是调度器唯一的调用入口，看它如何包装返回 `KVCacheBlocks` 对象。

### 4. 调度器与算子的解耦 (极简通信)
*   **`sched/output.py`**
    *   看 `class SchedulerOutput`: 亲自确认里面**没有**任何物理 block ids，只有极简的 `num_scheduled_tokens`。
*   (由于 v1 的 Worker 端代码较深，若想验证可以去) **`worker/model_runner.py`**
    *   追踪当收到 SchedulerOutput 时，Worker 是如何在内部唤醒自己的 `KVCacheManager`，利用状态机机制在本地重构 `block_tables` 的。