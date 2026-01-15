# `mha_triton.py` 为什么这么写：逐段讲清楚（学习版 Triton MHA Forward）

本文解释 `2026_0115/code_learning/mha_triton.py` 的实现思路：它实现的是 **Multi-Head Attention (MHA) 的 forward**（非 causal、无 dropout），并且用 **Triton 的“online softmax”** 技巧在计算时避免显式生成 \(S\times S\) 的 attention score 矩阵。

---

## 目标与张量形状约定

这份代码实现：

\[
O=\text{softmax}(QK^T\cdot \text{sm\_scale})V
\]

- **输入**：`q, k, v` shape 都是 **`[B, H, S, D]`**
  - \(B\)：batch
  - \(H\)：heads
  - \(S\)：seq length
  - \(D\)：head dim（也叫 head_size）
- **输出**：`o` 同 shape **`[B, H, S, D]`**
- `sm_scale` 默认是 \(1/\sqrt{D}\)，对应标准 attention 缩放。

参考实现（CPU/GPU 都能跑）在 `mha_torch_reference`：

```196:201:/mnt/d/github/learning/2026_0115/code_learning/mha_triton.py
def mha_torch_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, sm_scale: Optional[float] = None):
    B, H, S, D = q.shape
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * sm_scale, dim=-1)
    return torch.matmul(attn, v)
```

---

## 为什么不能直接算 `QK^T` 再 softmax？

直接算 `attn = softmax(QK^T)` 会构造一个 `[..., S, S]` 的矩阵：
- **显存/带宽开销大**：\(O(S^2)\) 存储与读写
- **性能差**：大量无谓的中间结果搬运

这也是 FlashAttention 这类实现的核心动机：**边算 score 边归一化边累积输出，不落地 score**。

本代码走的是简化版的 FlashAttention 思路：对每个 query block（`BLOCK_M` 行）在 K/V 上循环分块（`BLOCK_N` 列），用 online softmax 维护每行的 \(\max\) 与 \(\sum\exp\)。

---

## 并行映射：一个 Triton “program” 负责什么？

内核里用两个 `program_id`：

```41:44:/mnt/d/github/learning/2026_0115/code_learning/mha_triton.py
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh - b * H
```

含义：
- `pid_bh` 遍历 \(B\times H\) 个“头实例”（把 batch 和 head 打平）
- `pid_m` 沿序列维 \(S\) 切 query 的行块：每个 program 负责 `BLOCK_M` 个 query token

Python 侧 grid 配置也对应这个设计：

```159:161:/mnt/d/github/learning/2026_0115/code_learning/mha_triton.py
    grid = (triton.cdiv(S, BLOCK_M), B * H)
    mha_fwd_kernel[grid](
```

即：
- grid[0]：query block 数量 = \(\lceil S/BLOCK\_M\rceil\)
- grid[1]：\(B\times H\)

---

## 为什么 kernel 需要一堆 stride 参数？

Triton kernel 不认识 PyTorch 的“多维张量”，它只拿到指针 `q_ptr/k_ptr/v_ptr/o_ptr`。

为了从线性内存里定位 `q[b,h,m,d]`，需要把四维索引映射成地址：

\[
\text{addr}=q\_ptr + b\cdot stride\_{qb} + h\cdot stride\_{qh} + m\cdot stride\_{qm} + d\cdot stride\_{qd}
\]

你在代码里看到的就是这一套地址计算：

```52:60:/mnt/d/github/learning/2026_0115/code_learning/mha_triton.py
    q = tl.load(
        q_ptr
        + b * stride_qb
        + h * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd,
        mask=q_mask,
        other=0.0,
    ).to(tl.float32)
```

Python wrapper 把 `q.stride(i)` 传进去（PyTorch stride 单位是“元素”，不是字节，Triton 会按元素寻址）：

```166:173:/mnt/d/github/learning/2026_0115/code_learning/mha_triton.py
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
```

### 为什么 wrapper 里 `.contiguous()`？

```144:147:/mnt/d/github/learning/2026_0115/code_learning/mha_triton.py
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
```

这是“学习用”的简化：
- contiguous 后 stride 规律更简单，访问更连续，性能更稳定
- 也避免很多 stride/layout 角落问题

工程化时你可以不强制 contiguous，但要更小心地选择 tile、保证内存访问 coalesced，甚至按 layout 写不同 kernel。

---

## tile 的含义：`BLOCK_M/BLOCK_N/BLOCK_D`

在 kernel 内：
- `BLOCK_M`：一次处理多少个 query token（行数）
- `BLOCK_N`：一次加载多少个 key/value token（列块宽度）
- `BLOCK_D`：一次处理的 head_dim 宽度

本代码约束 `D <= BLOCK_D`（学习版，避免做 D 方向分块与累加）：

```154:157:/mnt/d/github/learning/2026_0115/code_learning/mha_triton.py
    BLOCK_D = 128 if D > 64 else 64
    if D > BLOCK_D:
        raise ValueError(f"head_dim D={D} 目前只支持 <= {BLOCK_D}（你可以把 BLOCK_D 调大或做分块）")
```

你会看到 `offs_d = tl.arange(0, BLOCK_D)`，并用 mask `offs_d < D` 来处理 `D` 小于 block 的情况：

```47:51:/mnt/d/github/learning/2026_0115/code_learning/mha_triton.py
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    q_mask_m = offs_m < S
    q_mask_d = offs_d < D
    q_mask = q_mask_m[:, None] & q_mask_d[None, :]
```

---

## 为什么要做 mask（`mask=` 和 `tl.where`）？

注意 \(S\) 和 \(D\) 通常不是刚好等于 block size 的倍数，因此最后一个 block 会越界：
- 越界 load/store 必须用 mask 防止非法访问
- 被 mask 的元素一般用 `other=0.0` 填充

`q/k/v` 的 load 都使用了 mask：

```52:91:/mnt/d/github/learning/2026_0115/code_learning/mha_triton.py
    q = tl.load(..., mask=q_mask, other=0.0).to(tl.float32)
    ...
    k = tl.load(..., mask=kv_mask, other=0.0).to(tl.float32)
    v = tl.load(..., mask=kv_mask, other=0.0).to(tl.float32)
```

此外 score 也要 mask：
- K 越界的位置应该是 \(-\infty\)，这样 softmax 权重为 0
- Q 越界的行要避免参与 `max/sum` 归约（否则会产生 NaN 或污染）

对应代码：

```93:99:/mnt/d/github/learning/2026_0115/code_learning/mha_triton.py
        scores = tl.dot(q, tl.trans(k)) * sm_scale
        scores = tl.where(kv_mask_n[None, :], scores, -float("inf"))
        scores = tl.where(q_mask_m[:, None], scores, -float("inf"))
```

---

## 核心：online softmax（为什么 `m_i / l_i / alpha` 这样写？）

对每一行 query（每个 token），softmax 需要：

\[
\text{softmax}(x)_j = \frac{e^{x_j}}{\sum_k e^{x_k}}
\]

直接算 `exp(x)` 会溢出，所以通常做数值稳定：

\[
e^{x_j - m},\quad m=\max(x)
\]

但这里 `x`（score）被分成了多个 K/V block（`BLOCK_N`）逐块处理。online softmax 维护两件事：
- `m_i`：当前处理到的 block 为止，这一行 score 的全局最大值
- `l_i`：当前处理到的 block 为止，这一行的 \(\sum \exp(score - m_i)\)

当新 block 来了，最大值可能变大，因此要把旧的“指数和/累积输出”按比例缩放到新的基准上，这就是 `alpha`：

```100:110:/mnt/d/github/learning/2026_0115/code_learning/mha_triton.py
        m_ij = tl.max(scores, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        p = tl.exp(scores - m_i_new[:, None])

        alpha = tl.exp(m_i - m_i_new)
        l_i_new = l_i * alpha + tl.sum(p, axis=1)

        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float32), v)
```

解释：
- `m_ij`：这一 block 内每行的最大 score
- `m_i_new`：更新后的全局最大值
- `p`：在新基准 `m_i_new` 下，这个 block 的“未归一化权重” \(e^{score-m_i_new}\)
- `alpha = exp(m_i - m_i_new)`：把旧 block 在旧基准下的结果缩放到新基准下
- `l_i_new`：更新后的指数和
- `acc`：更新后的加权和值（最终输出的分子）

最后再统一除以 `l_i` 完成归一化：

```112:115:/mnt/d/github/learning/2026_0115/code_learning/mha_triton.py
    denom = tl.where(q_mask_m, l_i, 1.0)
    out = acc / denom[:, None]
    out = tl.where(q_mask_m[:, None] & q_mask_d[None, :], out, 0.0)
```

这里 `denom` 对越界 Q 行用 1.0 是为了避免 `0/0 -> NaN`（即使后面 mask 掉，NaN 有时会影响调试/某些路径）。

---

## 为什么用 `float32` 计算/累加？

注意这些 `.to(tl.float32)`：

```52:65:/mnt/d/github/learning/2026_0115/code_learning/mha_triton.py
    q = ... .to(tl.float32)
    ...
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)
```

原因：
- softmax 对数值稳定更敏感，fp16/bf16 做 `exp/sum` 容易精度损失
- 累加 `p @ v` 属于 reduction，fp32 累加误差更小

输出再 cast 回输入 dtype（fp16/bf16）：

```117:125:/mnt/d/github/learning/2026_0115/code_learning/mha_triton.py
    tl.store(
        ...
        out.to(OUT_DTYPE),
        mask=q_mask,
    )
```

Python 侧根据输入 dtype 选择输出 dtype：

```159:160:/mnt/d/github/learning/2026_0115/code_learning/mha_triton.py
    out_dtype = tl.float16 if q.dtype == torch.float16 else tl.bfloat16
```

同时 wrapper 限制输入只能是 fp16/bf16（学习版先聚焦主流路径）：

```135:139:/mnt/d/github/learning/2026_0115/code_learning/mha_triton.py
    assert q.is_cuda and k.is_cuda and v.is_cuda
    ...
    assert q.dtype in (torch.float16, torch.bfloat16), "当前 Triton kernel 只支持 fp16/bf16 输入"
```

---

## `tl.dot` / `tl.trans` 在这里做了什么？

这一句是 score 计算：

```93:95:/mnt/d/github/learning/2026_0115/code_learning/mha_triton.py
        scores = tl.dot(q, tl.trans(k)) * sm_scale
```

此时：
- `q` shape: `[BLOCK_M, BLOCK_D]`
- `k` shape: `[BLOCK_N, BLOCK_D]`
- `tl.trans(k)` shape: `[BLOCK_D, BLOCK_N]`

所以得到 `scores` shape: `[BLOCK_M, BLOCK_N]`，也就是这一个 query block 对这一段 key block 的 attention logits。

输出累积使用：

```107:107:/mnt/d/github/learning/2026_0115/code_learning/mha_triton.py
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float32), v)
```

其中：
- `p` shape: `[BLOCK_M, BLOCK_N]`
- `v` shape: `[BLOCK_N, BLOCK_D]`
- `tl.dot(p, v)` -> `[BLOCK_M, BLOCK_D]`

这就是把该 block 的 V 按 softmax 权重加到输出分子上。

---

## wrapper 为啥这么写（调用方式/参数选择）

wrapper 做了几件关键事：

- **默认 sm_scale**：

```141:142:/mnt/d/github/learning/2026_0115/code_learning/mha_triton.py
        sm_scale = 1.0 / math.sqrt(D)
```

- **固定 tile**（学习用，后续可 autotune）：

```151:153:/mnt/d/github/learning/2026_0115/code_learning/mha_triton.py
    BLOCK_M = 64
    BLOCK_N = 64
```

- **warps 数量**：`num_warps=4` 是一个比较保守的选择，适合中等 tile。不同 GPU/shape 下最优值可能不同。

```191:191:/mnt/d/github/learning/2026_0115/code_learning/mha_triton.py
        num_warps=4,
```

---

## `__main__` 的测试为什么这样写？

### correctness

通过 PyTorch reference 对齐：

```225:229:/mnt/d/github/learning/2026_0115/code_learning/mha_triton.py
    out_triton = mha_triton_forward(q, k, v)
    out_ref = mha_torch_reference(q, k, v)

    max_diff = (out_triton - out_ref).abs().max().item()
    print(f"[check] max diff = {max_diff:.3e}")
```

### perf（非常简化）

预热 + 固定迭代计时：

```234:244:/mnt/d/github/learning/2026_0115/code_learning/mha_triton.py
    for _ in range(10):
        _ = mha_triton_forward(q, k, v)
    torch.cuda.synchronize()
    ...
    for _ in range(iters):
        _ = mha_triton_forward(q, k, v)
    torch.cuda.synchronize()
```

注意：这只是“有没有明显慢/快”的粗测。严谨 benchmarking 需要：
- `torch.cuda.Event` 或 Triton benchmark 工具
- 多组 `B/H/S/D`
- 固定随机种子、避免 CPU 干扰

### 为什么加了 “无 GPU 则只跑 CPU reference”

这是为了让脚本在没有 GPU 的机器上也能跑通，不至于直接断言退出：

```204:214:/mnt/d/github/learning/2026_0115/code_learning/mha_triton.py
    if not torch.cuda.is_available():
        print("[warn] torch.cuda.is_available() == False：当前环境没有可用 GPU，跳过 Triton，只运行 CPU reference。")
        ...
        raise SystemExit(0)
```

你有 GPU 时不会走这条分支。

---

## 这份实现的限制（你需要知道的边界）

- **非 causal**：没有 `i >= j` 的上三角 mask（自回归还不够用）
- **无 dropout / 无 attention bias / 无 alibi / 无 RoPE 等**（纯 MHA 核心公式）
- **只做 forward**：没有 backward（训练不可用）
- **`D <= 64/128` 的简化支持**：更大 `D` 需要在 D 维做分块累加
- **对 layout 简化**：强制 contiguous（便于学习与稳定表现）

---

## 怎么扩展（按优先级给你可落地的改法）

### 1) 加 causal mask（自回归）

在 block 内对 `scores` 再加一个 mask：当 `key_pos > query_pos` 时置为 `-inf`。

实现要点：
- `query_pos = offs_m[:, None]`
- `key_pos = offs_n[None, :]`
- `causal = key_pos <= query_pos`
- `scores = tl.where(causal, scores, -inf)`

同时要注意越界 mask 的组合（`kv_mask_n`、`q_mask_m`、`causal` 三者都要考虑）。

### 2) autotune（性能）

用 `@triton.autotune` 提供多组 `(BLOCK_M, BLOCK_N, num_warps)` 候选，key 用 `(S, D)` 或 `(S, D, H)`。

### 3) 支持更大 D / Dv 与 Dq 不同

需要把 `BLOCK_D` 拆成两类：
- score 的 dot 用 `Dq`
- `p@v` 的输出用 `Dv`

对大维度则按 D 分块，acc 在 D 维做累加（或用多个 program 处理 D tiles）。

### 4) KV cache / paged attention（偏 vLLM 场景）

K/V 不再是连续 `[S, D]`，而是块状分页。kernel 需要：
- 从 block table 映射物理页
- 按页加载 K/V
- 处理不同序列长度与 padding

---

## 结语

这份 `mha_triton.py` 的结构是典型的“学习版 FlashAttention 思路”：
- **grid** 切 query 行块 × (B*H)
- **for-loop** 扫过 K/V 列块
- 用 **online softmax（m/l/alpha）** 做数值稳定、避免落地 \(S\times S\)
- fp16/bf16 输入、fp32 计算/累加、最后 cast 回输出 dtype

如果你告诉我你最关心的扩展方向（比如 “我要 causal + KV cache” 或 “我要 D=128/256 并且更快”），我可以直接在现有文件上继续改到你需要的版本。

