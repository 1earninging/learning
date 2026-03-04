# vLLM(vllm-ascend) PCP/DCP & sglang CP/SP 讨论总结（2026-03-03）

> 本文整理本次对话中你提出的所有关键问题，并给出结论、实现位置与例子，便于后续复盘与面试表达。

---

### 1) vLLM(vllm-ascend) 里 PCP/DCP 是什么？分别在哪实现？

- **PCP（Prefill Context Parallelism）**
  - **目标**：把长 prompt 的 **prefill** 计算拆到多个 rank 并行做，最后把结果恢复到原始 token 顺序。
  - **Ascend 插件的核心实现**：`vllm-main/vllm-ascend-main/vllm_ascend/worker/pcp_utils.py` 的 `PCPManager`
    - 切分策略：**DualChunkSwap / 2*pcp_world_size 分块、头尾交错负载均衡**
    - 关键机制：pad/unpad mask、allgather 后 restore 索引、positions 重算等

- **DCP（Decode Context Parallelism）**
  - **目标**：把 **decode 阶段不断增长的 KV context** 在多个 rank 间交错分布，降低单卡 KV 内存占用；attention 输出需要跨 rank 归并（依赖 softmax LSE 等）。
  - **upstream vLLM 的关键实现点（在你仓库的 vLLM 主线里）**：
    - `vllm-main/vllm-main/vllm/v1/worker/gpu/cp_utils.py`：`prepare_dcp_local_seq_lens`（Triton kernel 计算本 rank 的 local seq lens）
    - `vllm-main/vllm-main/vllm/v1/worker/block_table.py`：`compute_slot_mapping`（用 mask 把非本 rank token 的 slot 写成 `-1`）
    - `vllm-main/vllm-main/vllm/v1/attention/backends/*`：decode metadata 构建与归并逻辑（如 FlashAttn/MLA/FlashInfer 等）

---

### 2) “PCP/DCP 怎么实现？”——核心数据流（Ascend 插件视角）

#### 2.1 PCP（prefill 拆分 + allgather/restore）

- **切分发生位置**：Ascend v1 runner 的 prepare_inputs 中（`vllm_ascend/worker/model_runner_v1.py` 调 `PCPManager.update_tokens_for_pcp`）
- **切分内容**：
  - per request 的 `num_scheduled_tokens` 会被改写为本 rank 需要处理的 token 数（prefill 被拆分）
  - positions 会按拆分后的 mapping 重算
- **恢复内容**：
  - prefill 结束后，hidden_states 会在 PCP group 上 **allgather**，并用 `pcp_allgather_restore_idx` 恢复原始顺序（以及配合 mask 去掉 pad token）

**DualChunkSwap 的示例**（来自 `PCPManager.update_tokens_for_pcp` 的 docstring）：

```text
tokens = [1, 5, 8], pcp_world_size = 2
pcp_rank = 0: pcp_tokens=[1,4,4], positions=[0,0,1,6,7,0,1,6,7]
pcp_rank = 1: pcp_tokens=[1,4,4], positions=[0,2,3,4,5,2,3,4,5]
```

#### 2.2 DCP（decode KV 交错存储 + 跨 rank 归并）

- **KV 分布（交错存储）**：
  - 通过 `BlockTable.compute_slot_mapping` 里的 mask 决定 token 属于哪个 rank（不属于就写 `-1`）
  - 粒度由 `cp_kv_cache_interleave_size` 控制（1 表示逐 token 轮转；更大表示以 interleave_size 为粒度轮转）
- **本 rank 的 local seq lens**：
  - upstream 会在 runner 侧计算 `dcp_local_seq_lens` 供 attention backend 用（避免过度 workspace 申请、避免 CPU sync）
- **为什么要求 softmax LSE**：
  - DCP decode 时每个 rank 只看到部分 KV，因此输出需要跨 rank 合并；合并通常需要 softmax 的稳定归一化信息（LSE）

---

### 3) “上层 scheduler 的 block manager 感知 PCP/DCP 么？”

结论（vLLM v1）：
- **Scheduler 侧会感知** `prefill_context_parallel_size` / `decode_context_parallel_size`，并把它们传入 `KVCacheManager`。
- 但这种“感知”主要体现在：
  - **KV block 的计量单位/对齐粒度/命中长度**按 `cp_world_size = pcp*dcp` 调整（block_size 会被放大到 cp 语义下的“虚拟块”）
  - 不负责 rank 级的切分与通信（这些在 worker/attention backend）

关键入口文件：
- `vllm-main/vllm-main/vllm/v1/core/sched/scheduler.py`：读取 pcp/dcp size 并创建 `KVCacheManager`
- `vllm-main/vllm-main/vllm/v1/core/single_type_kv_cache_manager.py`：在 `dcp*pcp>1` 时放大内部 `block_size`（影响 allocate/evict/cache-hit 计算）
- `vllm-main/vllm-main/vllm/v1/core/kv_cache_coordinator.py`：prefix cache hit 查找时携带 `dcp_world_size/pcp_world_size/alignment_tokens`

---

### 4) “vLLM 中 PCP/DCP 支持只开一个么？”

结论：
- **支持**只开其中一个。
- CLI 也分别提供开关：
  - `--prefill-context-parallel-size/-pcp`
  - `--decode-context-parallel-size/-dcp`

注意：同时启用时的 CP world 为 `pcp*dcp`，部分 backend/模型可能对 interleave、varlen、spec decode 等组合有限制（例如 interleave_size 与 MLA 的 varlen 支持条件）。

---

### 5) Ascend 插件：两个典型配置场景的说明

#### 5.1 `-pcp 2 -dcp 1`（只开 PCP）

- **主要收益**：长 prompt 的 **prefill** 更快（拆分计算、并行化）。
- **运行时关键变化**：
  - `PCPManager` 会把 prefill token 切给不同 PCP rank
  - prefill 结束后通过 PCP group allgather，并用 restore_idx 还原顺序
  - decode 阶段不启用 DCP 的 KV 交错与 LSE 归并（因为 dcp=1）

#### 5.2 `-pcp 1 -dcp 2`（只开 DCP）

- **主要收益**：**decode** 阶段 KV cache 的单卡内存压力下降（KV 在 rank 间交错分布），更容易跑更长上下文/更高并发。
- **运行时关键变化**：
  - slot_mapping 会按 `total_cp_world_size=2` 做 mask：非本 rank token 的 slot 为 `-1`
  - attention decode 需要跨 rank 归并（软归一化信息 LSE、all_to_all/all_gather 等）
  - prefill 不做 PCP 的 token 切分（pcp=1）

---

### 6) sglang 目录里 PCP/DCP 怎么实现？

结论：
- **sglang 没有沿用 vLLM 的命名 PCP/DCP**；它更偏向一个“attention context parallel（CP）”机制，且在你这份代码中明显聚焦于 **DeepSeek v3.2 的长 prefill（NSA）**：
  - `--attn-cp-size`：attention context parallel size
  - `--enable-nsa-prefill-context-parallel` + `--nsa-prefill-cp-mode`：NSA prefill CP 的切分模式（`in-seq-split` / `round-robin-split`）
- CP 的核心实现位置：
  - `sglang/sglang-main/python/sglang/srt/layers/attention/nsa/utils.py`
    - round-robin：按 `token_idx % cp_size` 分配 token
    - in-seq-split：按 `2*cp_size` 分块并 zigzag 重排以均衡负载
    - allgather 后按模式重排回原序

---

### 7) sglang 的调度层是否感知 CP / SP？

- **CP**：Scheduler 明确感知
  - 保存 `attn_cp_size/attn_cp_rank`，用于 DP-attention 的世界信息与通信分组
  - 在 DP-attention 模式下，会把 work 请求沿 attn-TP 再沿 attn-CP 广播
  - 调度策略层对 NSA prefill CP（in-seq-split）有额外限制：临时把 prefill batch 约束到 1（精度风险）

- **SP（你对话中确认的含义：Split Prefill）**：Scheduler 也感知
  - 通过 `ForwardMode.SPLIT_PREFILL` + `enable_pdmux` 进入专用的 forward 路径（PD multiplexing）

---

### 8) sglang：调度层“分配 block/page”时感知 CP 吗？

结论（在你这份 sglang 代码里）：
- **调度层在分配 KV cache 的 slot/page（out_cache_loc）时基本不感知 CP**。
- 解释：
  - 调度层确实会把请求广播到 CP 组（每个 CP rank 都会看到同一批 work），因此分配动作在各 rank 上是对称发生的。
  - `alloc_for_extend/alloc_token_slots` 等分配逻辑只按 `num_tokens/page_size` 工作，没有基于 `attn_cp_size` 做“只分配 local KV”的缩减。
  - CP 的主要作用发生在模型/attention forward（切分 token、allgather/restore），而不是 KV 分配层。

---

### 附：本次对话涉及的关键文件索引

- vLLM 主线（upstream）：
  - `vllm-main/vllm-main/vllm/v1/worker/block_table.py`
  - `vllm-main/vllm-main/vllm/v1/worker/gpu/cp_utils.py`
  - `vllm-main/vllm-main/vllm/v1/core/sched/scheduler.py`
  - `vllm-main/vllm-main/vllm/v1/core/single_type_kv_cache_manager.py`
  - `vllm-main/vllm-main/vllm/v1/attention/backend.py`
  - `vllm-main/vllm-main/vllm/engine/arg_utils.py`

- vLLM Ascend 插件：
  - `vllm-main/vllm-ascend-main/vllm_ascend/worker/pcp_utils.py`
  - `vllm-main/vllm-ascend-main/vllm_ascend/worker/model_runner_v1.py`
  - `vllm-main/vllm-ascend-main/vllm_ascend/attention/context_parallel/*.py`

- sglang：
  - `sglang/sglang-main/python/sglang/srt/server_args.py`
  - `sglang/sglang-main/python/sglang/srt/layers/attention/nsa/utils.py`
  - `sglang/sglang-main/python/sglang/srt/managers/scheduler*.py`
  - `sglang/sglang-main/python/sglang/srt/mem_cache/*`

