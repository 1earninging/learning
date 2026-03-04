# 2026_03_04 / code_learning

这里放一个 `prepare_cp_prefill_inputs` 的 **最小可运行 demo**，用来把“CP prefill + load balancing”里各类 index 的作用跑出来看清楚。

## 运行方式

在仓库根目录执行：

```bash
python3 /mnt/d/github/learning/2026_03_04/code_learning/demo_prepare_cp_prefill_inputs.py
```

该 demo **不依赖 torch/numpy**，纯 Python 可直接运行。

## demo 里你能看到什么

- `cp_load_balance_idx`：把 packed token 按“每个 request 前半 + 每个 request 后半”的方式重排（用于把 q/weights 的计算负载均衡化）。
- `cp_o_recover_idx`：把 “prev half 结果 + next half 结果” 拼起来后，再 gather 回原始 packed token 顺序。
- `cp_kv_recover_idx`：模拟 AllGather 后，把 KV 的顺序恢复成“正常时间顺序/chunk 顺序”。
- `actual_seq_lengths_query/key` + `k_gather_index_(prev/next)`：演示 indexer 计算 prev/next 两次时，key 侧如何按因果可见长度裁剪/抽取。
- **上游对齐**：Demo B 会先模拟 `text_generator` 在开启 CP 时的 **padding（对齐到 `2*cp_size`）** 与 **per-rank 切分（每个 cp_rank 取两段 chunk）**，再调用 `prepare_cp_prefill_inputs`，帮助你把“上游切分 → 下游索引”的全链路串起来。

> 注意：真实工程里 `past_key_states` 的内存布局、AllGather 的张量 shape 可能比 demo 更复杂；本 demo 的目标是“把索引生成与重排/恢复的概念跑通并可视化”，而不是完全复刻分布式通信细节。

