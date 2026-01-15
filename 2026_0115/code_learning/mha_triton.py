import math

import torch
import triton
import triton.language as tl
from typing import Optional


@triton.jit
def mha_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vn: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_om: tl.constexpr,
    stride_od: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    sm_scale,
    OUT_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh - b * H

    # queries: [BLOCK_M, BLOCK_D]
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    q_mask_m = offs_m < S
    q_mask_d = offs_d < D
    q_mask = q_mask_m[:, None] & q_mask_d[None, :]
    q = tl.load(
        q_ptr
        + b * stride_qb
        + h * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd,
        mask=q_mask,
        other=0.0,
    ).to(tl.float32)

    # online softmax state
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)

    # loop over K/V in blocks of N
    for start_n in range(0, S, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        kv_mask_n = offs_n < S

        # k: [BLOCK_N, BLOCK_D], v: [BLOCK_N, BLOCK_D]
        kv_mask = kv_mask_n[:, None] & q_mask_d[None, :]
        k = tl.load(
            k_ptr
            + b * stride_kb
            + h * stride_kh
            + offs_n[:, None] * stride_kn
            + offs_d[None, :] * stride_kd,
            mask=kv_mask,
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            v_ptr
            + b * stride_vb
            + h * stride_vh
            + offs_n[:, None] * stride_vn
            + offs_d[None, :] * stride_vd,
            mask=kv_mask,
            other=0.0,
        ).to(tl.float32)

        # scores: [BLOCK_M, BLOCK_N]
        scores = tl.dot(q, tl.trans(k)) * sm_scale
        # mask out invalid K positions
        scores = tl.where(kv_mask_n[None, :], scores, -float("inf"))
        # mask out invalid Q rows so they don't contaminate reductions
        scores = tl.where(q_mask_m[:, None], scores, -float("inf"))

        m_ij = tl.max(scores, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        p = tl.exp(scores - m_i_new[:, None])

        alpha = tl.exp(m_i - m_i_new)
        l_i_new = l_i * alpha + tl.sum(p, axis=1)

        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float32), v)

        m_i = m_i_new
        l_i = l_i_new

    # normalize
    denom = tl.where(q_mask_m, l_i, 1.0)
    out = acc / denom[:, None]
    out = tl.where(q_mask_m[:, None] & q_mask_d[None, :], out, 0.0)

    tl.store(
        o_ptr
        + b * stride_ob
        + h * stride_oh
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_od,
        out.to(OUT_DTYPE),
        mask=q_mask,
    )


def mha_triton_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, sm_scale: Optional[float] = None):
    """
    计算 MHA forward:
      out = softmax(q @ k^T * sm_scale) @ v

    约定输入输出 shape 为 [B, H, S, D]，且 q/k/v 的 D 相同。
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.ndim == 4 and k.ndim == 4 and v.ndim == 4
    assert q.shape == k.shape == v.shape
    assert q.dtype in (torch.float16, torch.bfloat16), "当前 Triton kernel 只支持 fp16/bf16 输入"
    B, H, S, D = q.shape
    assert D > 0 and S > 0
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    # 为了让 stride 更好看，强制 contiguous（学习用；工程里可更精细处理）
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    o = torch.empty_like(q)

    # 这里用固定的 tile，足够学习；后面你可以加 autotune
    BLOCK_M = 64
    BLOCK_N = 64
    # 要求 D <= BLOCK_D；常见 head_dim 64/128
    BLOCK_D = 128 if D > 64 else 64
    if D > BLOCK_D:
        raise ValueError(f"head_dim D={D} 目前只支持 <= {BLOCK_D}（你可以把 BLOCK_D 调大或做分块）")

    grid = (triton.cdiv(S, BLOCK_M), B * H)
    out_dtype = tl.float16 if q.dtype == torch.float16 else tl.bfloat16
    mha_fwd_kernel[grid](
        q,
        k,
        v,
        o,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        B=B,
        H=H,
        S=S,
        D=D,
        sm_scale=sm_scale,
        OUT_DTYPE=out_dtype,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=4,
    )
    return o


def mha_torch_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, sm_scale: Optional[float] = None):
    B, H, S, D = q.shape
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * sm_scale, dim=-1)
    return torch.matmul(attn, v)


if __name__ == "__main__":
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        print("[warn] torch.cuda.is_available() == False：当前环境没有可用 GPU，跳过 Triton，只运行 CPU reference。")
        B, H, S, D = 2, 4, 64, 64
        q = torch.randn((B, H, S, D), device="cpu", dtype=torch.float32)
        k = torch.randn((B, H, S, D), device="cpu", dtype=torch.float32)
        v = torch.randn((B, H, S, D), device="cpu", dtype=torch.float32)
        out_ref = mha_torch_reference(q, k, v)
        print(f"[cpu] out_ref mean={out_ref.mean().item():.6f}, std={out_ref.std().item():.6f}")
        raise SystemExit(0)

    device = "cuda"
    dtype = torch.float16

    # correctness（别把 S 设太大，torch reference 是 O(S^2)）
    B, H, S, D = 2, 4, 128, 64
    q = torch.randn((B, H, S, D), device=device, dtype=dtype)
    k = torch.randn((B, H, S, D), device=device, dtype=dtype)
    v = torch.randn((B, H, S, D), device=device, dtype=dtype)

    out_triton = mha_triton_forward(q, k, v)
    out_ref = mha_torch_reference(q, k, v)

    max_diff = (out_triton - out_ref).abs().max().item()
    print(f"[check] max diff = {max_diff:.3e}")

    # 简单计时
    import time

    for _ in range(10):
        _ = mha_triton_forward(q, k, v)
    torch.cuda.synchronize()

    iters = 50
    t0 = time.time()
    for _ in range(iters):
        _ = mha_triton_forward(q, k, v)
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"[perf] triton mha fwd: {(t1 - t0) * 1000 / iters:.3f} ms (B={B}, H={H}, S={S}, D={D})")

