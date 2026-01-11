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

### KV 传输细化视图（实现细节 / 边界条件）
- **vLLM：KV connector 数据流（Mooncake vs NIXL，host buffer/handshake/异构 TP）**：见 `vllm_kv_transfer_dataflow.puml`
- **vLLM：请求级 KV 传输状态机（async recv/send、超时、invalid blocks）**：见 `vllm_kv_transfer_state_machine.puml`
- **SGLang：PD disagg 数据流（chunk/page_indices/state_indices、prealloc/poll）**：见 `sglang_kv_transfer_dataflow.puml`
- **SGLang：请求级状态机（prefill/ decode 两端队列流转 + 失败回收）**：见 `sglang_kv_transfer_state_machine.puml`
- **对比：异构 TP、layout permute、host buffer、回收策略差异**：见 `compare_kv_transfer_edge_cases.puml`

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

---

## KV 传输：更细化的实现对齐（建议按这 4 个维度读）

### 维度 A：内存注册与“地址模型”（transfer backend 看到的是什么）
- **vLLM（kv_connector）**
  - Worker 侧通常会把 KV cache tensor（或 host_xfer_buffer）注册给后端，然后传输时按 **block_id -> (base_addr + block_id * block_len)** 计算 (src_ptr, dst_ptr)。
  - Mooncake：`batch_register_memory()` 注册 base pointers，发送时 `batch_transfer_sync_write()` 直接做批量写。
  - NIXL：register_memory + xfer descs + prep_xfer_dlist；当 device 内存不可直接注册或配置为 `kv_buffer_device=cpu` 时，走 **host_xfer_buffers + copy_blocks(d2h/h2d)**。
- **SGLang（disaggregation sender/receiver）**
  - 通过 `KVCache.get_contiguous_buf_infos()` 暴露 **每层 KV buffer 的 data_ptr/len/item_len** 给 transfer backend manager。
  - per-request 发送的核心不是“KV tensor 本体”，而是 **page_indices（以及最后一块可带 state_indices/metadata_buffer_index）**，后端把指定 pages 写入 decode 侧预分配区域。

### 维度 B：索引单位与映射（block_id vs page_indices vs loc）
- **vLLM**：以 `block_id` 为第一公民，BlockTable/slot_mapping 负责把 token position 映射到 KV 物理位置。
- **SGLang**：以 `loc`/`page_indices` 为第一公民，`ReqToTokenPool.req_to_token` 存 token->loc；`kv_to_page_indices()` 将 loc 映射为 page。

### 维度 C：异构 TP / KV layout / block_size mismatch 的处理策略
- **vLLM（NIXL）**
  - handshake 会交换 `kv_cache_layout`、`block_size`、`block_lens` 等；必要时启用 `enable_permute_local_kv` 在接收后做 layout 纠正（例如 HND<->NHD）。
  - 允许 “logical block_size != kernel block_size”，会在 worker 侧调整并做 logical->physical block id 映射（详见 NIXL 的 blocksize_post_process/physical block 逻辑）。
- **SGLang**
  - PD disagg 的核心是按 `page_size`（token-per-page）传输；chunk 发送时会把尾部不完整 page 延迟到下次发送（避免 partial page 语义复杂化）。
  - 对 hybrid 模型（SWA/Mamba/NSA）会额外发送 state_indices，保证 decode 侧能恢复必要的状态。

### 维度 D：生命周期与失败回收（谁负责释放/何时释放）
- **vLLM**：connector 允许“延迟释放 block”（request_finished 返回 delay_free）；worker 侧可能异步 send/recv，支持超时与 invalid block 上报。
- **SGLang**：prefill/inflight/transfer 队列轮询；成功时 `release_kv_cache` 解锁并回收 metadata buffer；失败则 abort 并显式释放/不插入缓存。


