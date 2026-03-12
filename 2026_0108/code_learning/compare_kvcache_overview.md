## SGLang vs vLLM：KVCache 管理与 KV 传输实现差异（总结 + PlantUML 视图）

本笔记基于本仓库当前代码（`sglang/`、`vllm/`）对 **KVCache 的内存管理**、**prefix KV 复用**、以及 **KV 传输（prefill↔decode / 跨实例）** 的实现做对齐式分析，并用 PlantUML 给出多视图设计图，方便快速建立整体心智模型与差异点定位。

> 重要说明：本仓库里的 vLLM 以 **v1 架构**为主（`vllm/vllm/v1/...`），SGLang 以 `sglang/python/sglang/srt/...` 为核心执行路径。

---

## 你应该记住的 4 个“核心差异”

- **差异 1：主索引单位不同**
  - **vLLM**：以 **block** 为主（`block_id`），围绕 `BlockPool`/`BlockTable` 做分配、共享、驱逐。
  - **SGLang**：以 **token loc/page** 为主（`ReqToTokenPool.req_to_token` 保存每个 token 的物理 loc），allocator 支持 `page_size>1` 的 page-aligned 分配与扩展/解码两类分配。

- **差异 2：prefix cache 的组织结构不同**
  - **vLLM**：以“**完整 block hash**”做 prefix cache 命中（hash 链：parent_hash + token_ids + extra keys），命中后增加 `ref_cnt`。
  - **SGLang**：以 **RadixTree（RadixCache/HiRadixTree）**做最长前缀匹配，支持在 node 内部命中时 **split**（page granularity 时对齐到 page）。

- **差异 3：KV 传输的抽象层次不同**
  - **vLLM**：提供独立的 `distributed/kv_transfer` 抽象（pipe / lookup buffer / connector），显式对接 scheduler/worker 的生命周期与异步加载/保存。
  - **SGLang**：PD-disaggregation 更偏“**两端共享/注册连续 KV 缓冲区** + per-request 只传 `page_indices`/元数据”，并可与 HiCache（L2/L3）叠加。

- **差异 4：跨实例复用与存储层次不同**
  - **vLLM**：prefix cache 主要在“单实例内的 GPU block pool”层；跨实例复用由 `kv_connector`/外部系统实现（Mooncake/NIXL 等）。
  - **SGLang**：HiCache 显式把 KV 分为 L1 GPU / L2 Host / L3 分布式存储，并提供 prefetch/write-back 策略与 page-first 等 I/O 友好布局。

---

## 多视图 PlantUML

### 组件视图（KVCache 管理与数据流）
- **vLLM v1：KVCache 管理组件图**：见 `vllm_kvcache_component.puml`
- **SGLang：KVCache + RadixCache/HiCache 组件图**：见 `sglang_kvcache_component.puml`
- **对比：两者关键模块对齐图**：见 `compare_kvcache_component_alignment.puml`

### 数据结构视图（索引/映射/元数据）
- **vLLM：BlockPool/BlockTable/BlockHash**：见 `vllm_kvcache_data_model.puml`
- **SGLang：ReqToTokenPool/Allocator/RadixCache/HiCache**：见 `sglang_kvcache_data_model.puml`

### 时序视图（请求生命周期：prefill→decode 与 KV 传输）
- **vLLM：disaggregated prefill + kv_connector（以 Mooncake/NIXL 为例）**：见 `vllm_kv_transfer_sequence.puml`
- **SGLang：PD disaggregation（prefill send chunk / decode prealloc+recv）**：见 `sglang_kv_transfer_sequence.puml`
- **对比：关键阶段对齐**：见 `compare_kv_transfer_sequence.puml`

---

## 代码锚点（读源码建议从这里开始）

### vLLM（v1）
- **KV cache 管理/分配**
  - `vllm/vllm/v1/core/kv_cache_manager.py`
  - `vllm/vllm/v1/core/kv_cache_coordinator.py`
  - `vllm/vllm/v1/core/block_pool.py`
  - `vllm/vllm/v1/worker/block_table.py`
  - `vllm/vllm/v1/core/kv_cache_utils.py`（block hash / extra keys / page size 统一等）
- **KV 传输（disaggregated prefill / 跨实例）**
  - `vllm/vllm/distributed/kv_transfer/README.md`
  - `vllm/vllm/distributed/kv_transfer/kv_connector/v1/`（`mooncake_connector.py`, `nixl_connector.py`）

### SGLang
- **KV cache 内存池**
  - `sglang/python/sglang/srt/mem_cache/memory_pool.py`（`ReqToTokenPool`, `KVCache`）
  - `sglang/python/sglang/srt/mem_cache/allocator.py`（paged allocator）
- **prefix cache / 分层 cache**
  - `sglang/python/sglang/srt/mem_cache/radix_cache.py`
  - `sglang/docs/advanced_features/hicache_design.md`
  - `sglang/python/sglang/srt/mem_cache/storage/`（Mooncake/NIXL/3FS/AIBrix 等）
- **PD disaggregation 传输路径**
  - `sglang/python/sglang/srt/disaggregation/prefill.py`
  - `sglang/python/sglang/srt/disaggregation/decode.py`


