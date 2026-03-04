# vLLM-Ascend：DeepSeek v3（MLA）在 Prefill 阶段的 CP（PCP/DCP）逐层逐域计算笔记

> 目标：把 **Prefill** 时 **每一层（Transformer layer）**、以及 **每个 CP 域（PCP/DCP rank）** 到底“算什么、怎么通信、KV 怎么落盘/拼接”的路径写清楚，便于复盘与面试表述。  
> 代码基于你仓库中的 `vllm-main/vllm-ascend-main` 与 `vllm-main/vllm-main`。

---

## 0. 术语与并行组（务必先对齐）

### 0.1 并行维度
- **TP (tensor parallel)**：切模型权重/头维，attention/MLP 中通常需要 all-reduce / all-gather。
- **PP (pipeline parallel)**：切层；本笔记默认 **PP=1**。
- **DP (data parallel)**：多个独立副本并行服务不同请求；本笔记重点是单个副本内部的 CP 逻辑。
- **PCP (prefill context parallel)**：prefill 阶段把 prompt token（Q/K/V 的 token 维）按策略拆到多个 rank。
- **DCP (decode context parallel)**：主要为 decode 的 KV 分片；但在 vLLM 的 CP 统一实现里，**slot_mapping/KV 落盘**对 prefill 也会按 `pcp*dcp` 的 “total CP world” 交错分配。

### 0.2 “CP 域”定义
把 **PCP + DCP** 合在一起看：
- `cp_world_size = pcp_world_size * dcp_world_size`
- `cp_rank = pcp_rank * dcp_world_size + dcp_rank`

token i（更准确是 positions 在 “虚拟 block” 内的 offset）最终会被映射到某个 `cp_rank`，其它 rank 对应 token 的 `slot_mapping=-1`（即不落到本 rank KV cache）。

**slot_mapping 计算**（Ascend Worker 版与 upstream v1 基本一致）见：
- `vllm_ascend/worker/block_table.py::BlockTable.compute_slot_mapping`
- `vllm/v1/worker/block_table.py::BlockTable.compute_slot_mapping`

其中关键逻辑：对每个 token，判断它是不是当前 `cp_rank` 的本地 token；不是则写 `-1`。

---

## 1. Prefill 的“外层时序”：一次 Engine step 里发生了什么

以 v1 runner + Ascend 插件为例，prefill 一次 step 的高层流程是：

1. **Scheduler 产出 `SchedulerOutput`**（包含每个 request 本 step 需要 prefill 的 token 数等）。
2. **ModelRunner 准备输入**（positions、input_ids、block_table、slot_mapping、attention metadata）：
   - `vllm_ascend/worker/model_runner_v1.py` 里：
     - 先按“未切分”算 positions、slot_mapping；
     - 若启用 CP/PCP：调用 `PCPManager.init_batch_info()` + `PCPManager.update_tokens_for_pcp()` 改写 `num_scheduled_tokens` 与 positions；
     - 构建 attention metadata 时调用：
       - `PCPManager.generate_pcp_metadata(...)`（生成 PCP 专用索引/序列长度信息）
       - `PCPManager.get_padded_slot_mapping(...)`（把 slot_mapping 扩到 PCP padding 后的长度；pad 位置为 -1）
3. **逐层执行 Transformer**（每层包含 attention + MLP 等）：
   - 注意：CP 主要影响 **attention 的 token 维计算与 KV cache 写入/通信**；MLP/Norm 等仍按 TP 常规通信。

本笔记后续重点解释第 2/3 步在 **DeepSeek v3 MLA** attention 上，CP 如何工作。

---

## 2. “PCP 切 token”在 prefill 的核心：DualChunkSwap + 两段 attention（mask/nomask）

### 2.1 PCP 的切分形态（直觉）
以 `pcp=2` 为例，一条长 prompt（prefill）会被切成 `2*pcp=4` 个 chunk，PCP rank 头尾交错拿 chunk，以均衡负载。

在 `PCPManager.update_tokens_for_pcp()` 的 docstring 里有示例（DualChunkSwap）。核心产物：
- 当前 PCP rank 实际需要处理的 token 数（`pcp_tokens`）
- 当前 rank 的 positions（`pcp_positions`）
- **拼回原序需要的 `pcp_allgather_restore_idx`**
- **pad/unpad mask**

见：`vllm_ascend/worker/pcp_utils.py::PCPManager.update_tokens_for_pcp`

### 2.2 为什么 token6 能用到 token1-5 的 KV（数学 + 工程）
对某个 query token（比如 token6）而言，正确的 causal attention 是：

$$o_6 = \sum_{j\le 6} \mathrm{softmax}(q_6^\top k_j)\, v_j$$

PCP 把 token 维拆开后，工程实现不会让 token6 “缺 KV 硬算”，而是把 KV 按对 token6 的可见性分成两类：

- **mask KV**：包含 token6 所在 chunk 的 KV，需要 **triu 因果 mask**（保证 token6 只看见 ≤6）
- **nomask KV**：严格早于 token6 的 KV（例如 token1-5），对 token6 来说不需要 mask

然后分别算两次 attention，并用 **LSE（log-sum-exp）** 做严格等价合并：

设两段 logits 指数和分别为 $Z_\text{mask}, Z_\text{nomask}$，局部输出分别为 $o_\text{mask}, o_\text{nomask}$，
全量输出等价于：

$$o = \frac{Z_\text{mask}}{Z_\text{mask}+Z_\text{nomask}}\,o_\text{mask}
     +\frac{Z_\text{nomask}}{Z_\text{mask}+Z_\text{nomask}}\,o_\text{nomask}$$

而 $Z$ 用 LSE 稳定计算：

$$\mathrm{lse}=\log(Z)=\log\big(e^{\mathrm{lse}_\text{mask}}+e^{\mathrm{lse}_\text{nomask}}\big)$$

这就是为什么“分段算 + 合并”在数学上与全量 softmax 完全等价（除浮点舍入误差）。

---

## 3. DeepSeek v3（MLA）在 Prefill：每层 attention 的 CP 计算链路

本节以 Ascend 的 MLA-CP 实现为准：
- `vllm_ascend/attention/context_parallel/mla_cp.py`
- PCP 索引生成：`vllm_ascend/worker/pcp_utils.py::PCPManager.generate_pcp_metadata`

### 3.1 每层 MLA attention 的输入张量（概念）
对每层 Transformer 的 attention：
- Query 侧来自上一层 hidden states（TP 切分后在本 TP rank 上持有部分头/权重）
- Key/Value 侧由当前 step 的 token 通过投影得到（MLA 的 kv 结构包含 `kv_c_normed + k_pe` 等），并写入 KV cache

CP 影响两个地方：
1) **本层 attention 计算时，KV 如何按“mask/nomask”分段参与**（保证因果 + 等价）
2) **KV cache 的写入/落盘**：slot_mapping 决定当前 CP rank 写哪些 token（其它 token slot=-1）

### 3.2 Prefill metadata：每个 CP 域要用到哪些索引/序列长度
`PCPManager.generate_pcp_metadata()` 会构造并塞进 `prefill_context_parallel_metadata`：
- `q_head_idx_tensor / q_tail_idx_tensor`：当前 PCP rank 的两段 query token 索引（head chunk / tail chunk）
- `kv_with_q_*_{mask,nomask}_idx_tensor`：对每段 query，该用哪些 KV 做 mask/nomask attention
- `attn_mask_seqlens / head_attn_nomask_seqlens / tail_attn_nomask_seqlens`：给 ring_mla 用的每段长度（支持多 request 拼 batch）
- `q_full_idx`：把 head/tail 两次算出来的输出拼回“本 rank 的 query 原顺序”

对应代码：
- 构造索引：`vllm_ascend/worker/pcp_utils.py::PCPManager.generate_pcp_metadata`（`q_head_idx/...` 那一大段）
- 在 MLA CP metadata builder 中取出：`vllm_ascend/attention/context_parallel/mla_cp.py` 顶部的 `build_cp_metadata(...)`

### 3.3 每层：KV 的生成、PCP all_gather/restore、以及 KV cache 写入（按 CP 域）
当 `pcp_size>1` 时，MLA prefill 的 preprocess 会显式做：
1) 从本 rank 的 token 生成局部 `kv_c_normed` 与 `k_pe`；
2) **在 PCP group 上 all_gather** 把各 PCP rank 的 KV 片段收集起来；
3) 用 `pcp_allgather_restore_idx` **重排回原 token 顺序**；
4) 用 `slot_mapping` 把 KV 写入本地 KV cache：**只有 slot!=-1 的 token 会真正落盘**；

见：`vllm_ascend/attention/context_parallel/mla_cp.py::mla_preprocess_prefill`
- `get_pcp_group().all_gather(...)`
- `torch.index_select(..., pcp_allgather_restore_idx)`
- `torch_npu._npu_reshape_and_cache(..., slot_indices=slot_mapping)`

> 注意：这里的 all_gather/restore 主要是为了把 KV 张量整理成“可用于后续 attention/落盘”的顺序；  
> KV cache 是否在每个 rank 都“完整复制一份”，取决于 `slot_mapping`。在 CP>1 时，slot_mapping 会对非本地 token 写 -1，因此 **KV cache 在 CP ranks 之间是分片/交错分布的**。

### 3.4 每层：attention 本体（对每个 CP 域，算哪些 Q、看哪些 KV）
当 `pcp_size>1` 时，每层 prefill attention 会走 `_forward_prefill()` 的 PCP 分支：

1) 取出 PCP metadata 的索引：
   - `q_head_idx / q_tail_idx`
   - `kv_with_q_head_* / kv_with_q_tail_*`
2) 对 head chunk 与 tail chunk **分别调用** `_attention_with_mask_and_nomask(...)`：
   - **mask 部分**：`npu_ring_mla(mask_type_triu)`，使用 `kv_mask_idx` 对应的 KV（需要 causal）
   - **nomask 部分**：对 `kv_nomask_idx`（可能被 split 成多段）循环调用 ring_mla，并用 `pre_out/prev_lse` 递推合并（LSE 合并严格等价）
3) 拼回输出顺序：`q_full_idx` 把 head/tail 的输出重新排成当前 rank 的 query 顺序

见：`vllm_ascend/attention/context_parallel/mla_cp.py::AscendMlaCPImpl._forward_prefill`
以及 `_attention_with_mask_and_nomask(...)` 内部对 `torch_npu.atb.npu_ring_mla(...)` 的两段调用逻辑。

> 直觉对应你提的例子：token6 若落在 tail chunk，它的 nomask KV 会包含 token1-5；mask KV 包含 token6 所在 chunk 的局部 KV 并用 triu mask 保证因果。两段通过 LSE 合并得到与全量 attention 等价的结果。

---

## 4. 每个 CP 域到底“存了哪些 KV”（slot_mapping 视角）

### 4.1 slot_mapping 如何决定“本域 token”
无论是 Ascend 版还是 upstream v1，逻辑基本一致：

1) 定义虚拟 block：
   - `virtual_block_size = block_size * cp_world_size`
2) 在虚拟 block 内根据 `cp_kv_cache_interleave_size` 轮转分配 token：
   - 满足条件的 token 才会落在当前 `cp_rank`
3) `slot_mapping = block_id * block_size + local_offset`，非本域 token 写 `-1`

对应（upstream v1 版更简洁易读）：
- `vllm/v1/worker/block_table.py::BlockTable.compute_slot_mapping`

因此，在 CP>1 的情况下：
- **每个 CP rank 的 KV cache 只存自己负责的 token**
- 其它 token 通过 `slot_mapping=-1` 不写入本地 cache

### 4.2 这对后续阶段意味着什么
prefill 的 attention 之所以能“看到全量历史 KV”，是因为实现会在需要时做 CP 维度的收集/重排（如 PCP all_gather/restore），或者在 decode 用 DCP 的 out+LSE 归并等方式得到等价输出。  
**KV cache 本身不一定在每个 rank 都完整复制**——这正是 CP/DCP 降内存的核心。

---

## 5. Chunked Prefill（可选）：CP 域如何按 chunk 处理长上下文

当启用 chunked prefill 且上下文很长时，MLA-CP 会构造 `CPChunkedContextMetadata`，其中：
- `local_context_lens_allranks`：每个 request 在每个 CP rank 上“本地拥有的上下文长度”
- `padded_local_chunk_seq_lens` / `padded_local_cu_seq_lens`：把 chunk 切成适配 workspace 的分块

见：`vllm_ascend/attention/context_parallel/mla_cp.py` 中 `build_chunked_metadata(...)`。

其目的：让每个 CP rank 只处理自己那份 KV 的 chunk（并通过必要的 gather/reorg 适配 kernel 布局），避免一次性为超长序列申请过大 workspace。

---

## 6. 一段“按层/按域”的可复述话术（面试版）

对 DeepSeek v3 的 MLA，在 prefill 阶段启用 CP（PCP/DCP）后：
- **每层 attention** 的 Q/K/V 投影仍按 TP 常规切分；
- **token 维的并行**由 PCP/DCP 控制：
  - `slot_mapping` 把 token 的 KV 写入分散到不同 CP rank（非本域 token slot=-1）；
  - 为保证因果与等价性，prefill attention 会把 KV 按 **mask/nomask** 拆段，并用 **LSE 合并**得到与全量 softmax 一致的输出；
  - PCP 场景下会构造 `q_head/q_tail` 与 `kv_with_q_*` 等索引，使得每个 PCP rank 在本层内只对自己那两段 query 做 ring_mla，两段输出再用 `q_full_idx` 拼回顺序。

---

## 7. 关键实现文件索引（便于回看）

- **PCP 切分与索引/metadata**：
  - `vllm-main/vllm-ascend-main/vllm_ascend/worker/pcp_utils.py`
- **slot_mapping / KV 分片落盘**：
  - `vllm-main/vllm-ascend-main/vllm_ascend/worker/block_table.py`
  - `vllm-main/vllm-main/vllm/v1/worker/block_table.py`
- **MLA + CP 的 prefill preprocess/attention**：
  - `vllm-main/vllm-ascend-main/vllm_ascend/attention/context_parallel/mla_cp.py`
- **runner 输入准备与 metadata 注入**：
  - `vllm-main/vllm-ascend-main/vllm_ascend/worker/model_runner_v1.py`



---

## 8. 当算子不支持 LSE：一种“PCP 不依赖 LSE”的实现策略（KV 同步 + 切 Q）

> 这一节解释：如果 attention kernel **拿不到 softmax LSE（或其它等价统计量）**，PCP 仍然可以做到数学严格等价的一种实现路线。

### 8.1 关键思想

当你把 KV 分到多个 rank 上各算一部分 attention 时，想要把局部输出严格合并成全量输出，必须要有归一化信息（典型是 LSE，或 max+sumexp）。

如果 kernel 不提供 LSE，你可以选择另一条路线：

- **先把本层需要的 KV 同步/重建成“全量可见 KV”**（在 PCP group 内 all-gather + reorder/restore），使得每个 PCP rank 都能看到相同的可见 key/value。
- **再只在 Q 维度做并行**：每个 PCP rank 只计算自己负责的那段 query token，但每个 query 的 attention 仍是“对全量可见 KV 的一次性 softmax”。
- 最后把各 rank 的输出按 token 顺序拼回即可。

这条策略本质上让 PCP 退化为 **query-parallel**（切 Q），避免了跨 rank softmax 合并。

### 8.2 数学等价性（为什么不需要 LSE）

标准 causal attention（单头写法）：

$$o_i = \sum_{j \le i} \mathrm{softmax}(s_{ij})\, v_j,\quad s_{ij} = q_i^\top k_j + m_{ij}$$

在“KV 同步 + 切 Q”策略下：

- 对任意 token $i$，它在某一个 rank 上被计算一次；
- 该 rank 拥有计算 $o_i$ 所需的全部 $\{k_j,v_j\}_{j\le i}$；
- softmax 的分母 $\sum_{j\le i} e^{s_{ij}}$ 在一次 kernel 内完整计算。

因此输出 $o_i$ 与不切分、单卡全量计算严格一致（仅受浮点舍入影响），不需要 LSE。

### 8.3 工程代价

- **优点**：实现简单、严格等价、不依赖 LSE。
- **缺点**：每层都要同步较大体积的 KV（PCP all-gather），通信/显存开销可能显著上升，`pcp_size` 越大越明显。

---

## 9. 最近问题 Q&A 汇总（4 问）

### 9.1 如果不是 ring-MLA，两个 CP 的 attention 结果怎么汇总？

- **核心**：每个 CP rank 先算局部的 $(out_r,\mathrm{lse}_r)$（或等价统计量），再严格等价合并：
  - $\mathrm{lse}=\log\sum_r e^{\mathrm{lse}_r}$
  - $out=\sum_r e^{\mathrm{lse}_r-\mathrm{lse}}\, out_r$
- **工程**：可用 all-gather / all-to-all 收集 `out+lse`，再用 update 算子（例如 `npu_attention_update`）做合并。

### 9.2 如果算子无法支持 LSE 呢？

- **无法**用“一趟局部 softmax 输出 + LSE 合并”的方式做严格等价归并。
- **精确替代**：做分布式 softmax 的两趟统计（或融合实现）：
  - all-reduce(max) 得到全局 $m$
  - all-reduce(sum) 得到 $\sum e^{s-m}$ 与 $\sum e^{s-m}v$，输出为 $out = \frac{\sum e^{s-m}v}{\sum e^{s-m}}$
- **退化**：把 KV/logits gather 到单处再算全量 attention（通信/显存代价大），或做近似合并（不等价）。

### 9.3 不支持 LSE，PCP 和 DCP 都会受影响吗？是不是只影响 DCP，不影响 PCP？

- **DCP 基本必受影响**：因为它的目标就是 KV 分片省内存，必须做跨 rank softmax 归一化合并（需要 LSE 或 max/sumexp）。
- **PCP 是否受影响取决于策略**：
  - 如果 PCP 走“分段 attention + LSE 合并”（例如 vLLM-Ascend 的 DeepSeek v3 MLA PCP），则 PCP 也受影响；
  - 如果 PCP 改走第 8 章的“KV 同步 + 切 Q”，则 PCP 可不依赖 LSE。

### 9.4（本问题）PCP 不受影响的策略如何实现？数学上如何等价？

- **实现**：每层先 PCP all-gather+restore 得到全量可见 KV，再按 Q 切分让各 rank 各算一段 query 的“完整 causal attention”，最后拼接输出。
- **等价性**：每个 query 的 softmax 分母仍在一次 kernel 内覆盖所有可见 KV，因此与全量 attention 严格一致。

---

## 10. 补充：MindIE-LLM 的 CP Prefill（SparseFusionAttention）整体调用链路与逻辑（不依赖 LSE）

> 对应文件：`MindIE-LLM/mindie_llm/runtime/layers/attention/backend/sparse_attention.py`

这一节把你贴的 `prepare_cp_prefill_inputs` 产物（`cp_load_balance_idx/cp_kv_recover_idx/cp_o_recover_idx` 等）在**真实调用路径**里如何被使用逐点串起来。

### 10.1 这套 CP-prefill 的核心结论

- 它不是“每个 CP rank 只持有部分 KV → 局部 attention → 用 LSE 合并”。
- 它更像“**先把 KV/rope 相关输入通过 allgather+reorder 整理好**，然后对 Q 做 load-balance 重排，拆成 prev/next 两段分别跑 attention，最后用 recover_idx 拼回输出”。
- 因此它不依赖 attention kernel 输出 LSE 来做跨 CP 合并。

### 10.2 CP 输入从哪里来：`SfaMetadataBuilder.build → prepare_cp_prefill_inputs`

- `SfaMetadataBuilder.build(...)` 在 **`cp_size>1` 且 `is_prefill`** 时调用：
  - `prepare_cp_prefill_inputs(cp_size, input_ids, position_ids, actual_seq_lengths_query, actual_seq_lengths_kv)`
- 生成的 `cp_input_dict` 被放入 `SfaMetadata(cp_input_dict=...)`，随后在每层 forward 中可用。

`prepare_cp_prefill_inputs` 里关键索引含义：

- **`cp_load_balance_idx`**：把每个 request 的 token 切成前半/后半，并把“所有 request 的前半”拼一起、“所有 request 的后半”拼一起。
- **`cp_o_recover_idx`**：把 prev/next 两段输出拼回原 token 顺序。
- **`cp_kv_recover_idx`**：KV allgather 后恢复正常 token 顺序。
- **`k_gather_index_prev/next` + `actual_seq_lengths_key/query`**：为 prev/next 两段 attention 提供各自需要的 KV 子集与变长信息。

### 10.3 每层 forward 的实际调用链路（按代码顺序）

1) **进入 attention**：`SfaBackendImpl.forward(layer, hidden_states, kv_cache)`
- 从 `forward_context.attn_metadata_dict[layer.prefix]` 取 `SfaMetadata`（里面可能有 `cp_input_dict`）。

2) **预处理**：`sfa_preprocess(...)` 在 prefill 时调用 `sfa_prefill_preprocess(...)`

3) **Prefill preprocess 做 KV/rope 的 allgather+reorder**（使用 `cp_kv_recover_idx`）
- 在 `sfa_prefill_preprocess` 里，prefill 且 `cp_size>1`：
  - `kv_no_split = allgather_and_reorder(kv_no_split, cp_group, cp_kv_recover_idx)`
  - `cos/sin = allgather_and_reorder(cos/sin, cp_group, cp_kv_recover_idx)`
  - 然后调用 `npu_kv_rmsnorm_rope_cache(..., is_output_kv=True)` 写 cache 并返回用于后续的 key_states。

4) **Indexer（稀疏 topk）也对齐 CP 切分**：`indexer_select(...)`
- prefill 且 `cp_size>1`：
  - 先 `k = allgather_and_reorder(k, ..., cp_kv_recover_idx)`
  - 再 `q/weights = gather_tensor(..., cp_load_balance_idx)`
  - `do_npu_cp_balance_indexer` 里按 prev/next 做两次 indexer
  - 最后 `topk_indices = gather_tensor(topk_indices, cp_o_recover_idx)` 恢复回原序。

5) **真正的 CP prefill attention**：`apply_prefill_sfa → do_cp_balance_attn`
- `do_cp_balance_attn` 的关键步骤：
  - `q_nope/q_pe/topk_indices = gather_tensor(..., cp_load_balance_idx)`（负载均衡重排）
  - split 成 prev/next
  - `k_nope/k_pe = gather_tensor(k, k_gather_index_prev/next)`（为每段 query 选取相应 key 子集）
  - 分别调用两次 `torch_npu.npu_sparse_flash_attention(...)` 得到 `attn_out_prev/attn_out_next`
  - `attn_out = cat(prev,next)`
  - `attn_out = gather_tensor(attn_out, cp_o_recover_idx)`（恢复回原 token 顺序）

### 10.4 为什么这里不需要 LSE

- 对每次 `npu_sparse_flash_attention` 调用而言，key/value 已经通过 `allgather_and_reorder` + `k_gather_index_prev/next` 被准备成“该段 query 所需的完整集合”，softmax 归一化在 kernel 内完成。
- 因此最终只需要做 **输出的索引重排恢复**（`cp_o_recover_idx`），而不是跨 CP rank 做 softmax 合并。
- 这也解释了为什么代码可以写成 `attn_out_prev, _, __ = ...`：额外返回值即使包含 LSE，也不是这条 CP 合并路径所必需。

