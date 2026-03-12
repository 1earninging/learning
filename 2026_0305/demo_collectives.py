import argparse

import torch
import torch.distributed as dist


def init(backend: str):
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world = dist.get_world_size()
    return rank, world


def reduce_scatter_sum_fallback(big: torch.Tensor, world: int) -> torch.Tensor:
    """
    兼容后端：当某些后端（例如部分版本的 gloo）不支持 reduce_scatter 时，
    用 all_reduce(big) + 按 rank 切片 的方式模拟 reduce_scatter(sum)。
    语义等价，但性能不代表真实 reduce_scatter。
    """
    if big.numel() % world != 0:
        raise ValueError(f"big.numel()={big.numel()} must be divisible by world={world}")

    dist.all_reduce(big, op=dist.ReduceOp.SUM)
    shard = big.chunk(world)[dist.get_rank()].contiguous()
    return shard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="gloo", choices=["gloo", "nccl"])
    args = parser.parse_args()

    rank, world = init(args.backend)
    device = torch.device("cuda", rank) if args.backend == "nccl" else torch.device("cpu")

    # 每个 rank 放一个不同的张量，便于观察
    x = torch.ones(4, device=device) * (rank + 1)

    # 1) AllReduce(sum)
    y = x.clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    if rank == 0:
        print("[all_reduce sum] expected =", sum(range(1, world + 1)), "got", y[0].item())

    # 2) AllGather
    gather_list = [torch.empty_like(x) for _ in range(world)]
    dist.all_gather(gather_list, x)
    z = torch.cat(gather_list, dim=0)
    if rank == 0:
        print("[all_gather] got", z.tolist())

    # 3) ReduceScatter(sum)
    # 输入需要是输出分片大小的 world 倍：这里用长度 4*world
    big = torch.arange(4 * world, device=device, dtype=torch.float32) + rank
    out = torch.empty(4, device=device, dtype=torch.float32)
    try:
        dist.reduce_scatter(out, list(big.chunk(world)), op=dist.ReduceOp.SUM)
    except RuntimeError as e:
        # 常见：ProcessGroupGloo does not support reduce_scatter
        # 学习语义时，用 all_reduce + slice 做等价模拟
        out.copy_(reduce_scatter_sum_fallback(big, world))
        if rank == 0:
            print(f"[reduce_scatter] fallback used due to: {type(e).__name__}: {e}")
    if rank == 0:
        print("[reduce_scatter] rank0 out", out.tolist())

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

