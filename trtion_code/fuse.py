import torch
import triton
import triton.language as tl

@triton.jit
def fuse_kernel(x_ptr, y_ptr, z_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)                       # 当前“程序块”id（沿着第0维）
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # 本块负责的一批 index
    mask = offsets < n_elements                  # 最后一块可能越界，用 mask 截断

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    z = tl.load(z_ptr + offsets, mask=mask)
    out = x * y + z
    tl.store(out_ptr + offsets, out, mask=mask)

def fuse(x, y, z, BLOCK_SIZE=1024):
    assert x.is_cuda and y.is_cuda and z.is_cuda
    assert x.shape == y.shape
    assert x.shape == z.shape

    n_elements = x.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    out = torch.empty_like(x) 
    
    fuse_kernel[grid](x, y, z, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out

if __name__ == "__main__":
    device = "cuda"
    N = 1 << 24  # 约 1600 万个元素，稍微大一点才看得出性能差异
    x = torch.randn(N, device=device, dtype=torch.float32)
    y = torch.randn(N, device=device, dtype=torch.float32)
    z = torch.randn(N, device=device, dtype=torch.float32)

    def run_and_check(BLOCK_SIZE):
        # 预热
        for _ in range(10):
            out_triton = fuse(x, y, z, BLOCK_SIZE=BLOCK_SIZE)
        torch.cuda.synchronize()

        # 计时
        import time
        iters = 50
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            out_triton = fuse(x, y, z, BLOCK_SIZE=BLOCK_SIZE)
        torch.cuda.synchronize()
        t1 = time.time()

        out_torch = x * y + z
        max_diff = (out_triton - out_torch).abs().max().item()
        print(f"BLOCK_SIZE={BLOCK_SIZE}, max diff={max_diff:.3e}, "
              f"time={(t1 - t0)*1000/iters:.3f} ms")

    for bs in [32, 64, 128, 256, 512, 1024, 2048]:
        run_and_check(bs)