## 目标与学习路径（建议 1-2 天入门，1 周熟练）

- **你要掌握的核心**：集体通信（Collective Communication）里最常见的几类操作（AllReduce / AllGather / ReduceScatter / Broadcast / Reduce / Gather / Scatter / AllToAll），它们“语义是什么、代价大概多少、为什么这么慢/这么快、怎么在框架里用、怎么排查性能问题”。
- **推荐学习顺序**：
  - **先理解语义**：每个 collective 的输入/输出张量是什么关系。
  - **再看典型算法**：ring / tree / hierarchical（分层：机内 + 机间）。
  - **最后落到实践**：用 PyTorch Distributed（NCCL）跑通，再用 profiler 看瓶颈。

---

## 0. 背景：为什么会有这些通信操作

分布式训练里最典型的是数据并行（DP）：每张卡算一份梯度，最后需要把梯度在所有卡之间“合并一致”。这一步就是 **AllReduce(grad)**。

而在张量并行/专家并行（TP/EP）中，常见模式是“把各卡上的一部分结果拼起来”，这往往是 **AllGather** 或 **AllToAll**。

---

## 1. 术语与记号（后文统一口径）

- **world_size = P**：进程/设备总数。
- **rank**：当前进程编号 $0..P-1$。
- **N**：每个 rank 上参与通信的数据字节数（或元素数）。
- **带宽（Bandwidth）B**：链路可持续吞吐（例如 NVLink、PCIe、IB）。
- **时延（Latency）α**：每次通信启动/同步的固定成本。
- **简化代价模型**：常见写法为 $T \approx \alpha \cdot \#steps + \beta \cdot \#bytes$，其中 $\beta = 1/B$。

---

## 2. 语义速查表（最重要：先把“做了什么”搞清楚）

下面假设每个 rank 上都有同形状张量 `x_r`。

- **Broadcast (Bcast)**：某个 root 把自己的 `x_root` 发送给所有人，所有 rank 最终都得到同一个 `x_root`。
- **Reduce**：把所有 `x_r` 用 op（sum/max/avg）聚合到 root：`y_root = op_r x_r`。
- **AllReduce**：所有 rank 都得到聚合结果：`y_r = op_k x_k`（对所有 r 都一样）。
- **Gather**：root 收集每个 rank 的 `x_r` 拼成更大张量 `Y_root = concat_r x_r`。
- **AllGather**：每个 rank 都得到拼好的大张量：`Y_r = concat_k x_k`。
- **Scatter**：root 把大张量切分成 P 份分发给各 rank。
- **ReduceScatter**：先 reduce 再 scatter（或等价的融合算法）：每个 rank 得到聚合结果的一段。
- **AllToAll**：每个 rank 把自己数据切 P 份，分别发给其它 rank；最终每个 rank 收到来自所有人的一份（典型于 MoE 的 token/专家路由）。

**一句话记忆**：
- **AllReduce**：大家“求和/求平均”，最后“人人一份相同结果”。
- **AllGather**：大家“拼接”，最后“人人一份拼好的大结果”。
- **ReduceScatter**：大家“求和/求平均”，但最后“每人只拿自己那一段”（常用于优化带宽）。

---

## 3. AllReduce：最常用的操作

### 3.1 典型算法：Ring AllReduce（带宽友好）

Ring AllReduce 通常分两段：

- **Reduce-Scatter（P-1 步）**：每步每个 rank 发一段、收一段并做 reduce，最终每个 rank 只保留聚合后的 1/P 数据块。
- **AllGather（P-1 步）**：再把这 1/P 的块互相广播式拼回全量。

**总通信字节（每个 rank 视角）**：
- 发送：约 $2 \cdot (P-1)/P \cdot N$
- 接收：同量级

**为什么 ring 常见**：它在大消息场景下接近带宽上限，扩展性好；缺点是小消息时延开销明显。

### 3.2 Tree AllReduce（时延更好）

二叉树/多叉树 reduce + broadcast：
- 步数大约 $O(\log P)$，更适合小消息/高时延网络。
- 但链路利用率可能不如 ring（依实现而定）。

### 3.3 实战建议（训练里最常见）

- **大梯度张量（MB 级）**：ring/hierarchical 通常更合适。
- **大量小张量**：最怕 α（启动时延）。常见优化：
  - **梯度 bucket**（PyTorch DDP 默认会把梯度合并成桶再 allreduce）
  - **融合通信 kernel**（NCCL/厂商库）

---

## 4. AllGather：拼接类通信（TP/序列并行很常见）

AllGather 的输出大小是输入的 **P 倍**：每个 rank 最终都持有 `P*N` 数据（按元素数计）。

常见使用场景：
- **张量并行**：每卡算一部分输出通道，最后 allgather 拼完整 hidden。
- **序列并行**：把 sequence 维分片后，需要在某些层/算子前后 gather。

关键性能点：
- 因为输出放大，**显存/带宽压力大**；有时会改成 **ReduceScatter + matmul** 的形式减少峰值。

---

## 5. ReduceScatter：AllReduce 的“更省带宽/更友好算子融合”版本

在一些并行策略里，不一定需要“每卡都拿到完整 reduce 结果”，而是后续算子只用其中一段：
- 用 **ReduceScatter** 可以把 “reduce + 分发”融合，通常更高效。
- 典型组合：**ReduceScatter + AllGather**（类似 ring allreduce 的两段），也常用于 TP 的前后向通信。

---

## 6. AllToAll：MoE/路由类的核心（最容易踩坑）

AllToAll 的特点：
- 每卡既当 sender 又当 receiver，数据量取决于切分策略与负载均衡。
- 代价受 **不均匀（skew）** 影响很大：如果某些 rank 收到/发送更多，会拖慢整体（慢者为王）。

常见优化：
- **容量因子（capacity factor）**、top-k 路由约束、token 重排（balance）。
- **分层 alltoall**（机内先交换、再跨机交换）。

---

## 7. 代价模型与直觉（你用它来“估算是不是通信瓶颈”）

### 7.1 粗略估算（够用就行）

- **Ring AllReduce**：$T \approx 2(P-1)\alpha + 2(P-1)/P \cdot N/B$
- **Ring AllGather**：$T \approx (P-1)\alpha + (P-1)/P \cdot N/B$（这里 N 指最终全量大小或按实现定义；核心是“每步搬运 1/P”）

直觉：
- **小消息**：$\alpha$ 主导，步数越多越吃亏（ring 相对 tree 更差）。
- **大消息**：$N/B$ 主导，ring 往往更接近带宽上限。

### 7.2 性能指标你应该看什么

- **链路带宽利用率**：实际吞吐 / 理论峰值。
- **通信-计算重叠**：通信是否能隐藏在反向计算后面（DDP overlap）。
- **bucket size**：桶过小 -> α 过多；过大 -> overlap 变差/显存峰值上升。

---

## 8. PyTorch 实操（最小可运行例子）

下面示例可以在单机多卡（或多进程）环境跑通语义。若你现在没有多 GPU，也可以用 CPU 的 `gloo` 后端验证。

### 8.1 运行方式（示例）

- 单机 4 进程（CPU/gloo）：

```bash
torchrun --standalone --nproc_per_node=4 demo_collectives.py
```

- 单机多 GPU（NCCL）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 demo_collectives.py --backend nccl
```

### 8.2 示例代码（保存为 `demo_collectives.py` 自行运行）

> 说明：有些 PyTorch 版本里 **`gloo` 后端不支持 `reduce_scatter`**（会报 `ProcessGroupGloo does not support reduce_scatter`）。
> 这种情况下可以用 “`all_reduce` + 按 rank 切片” 来**等价模拟语义**（但性能不代表真实 `reduce_scatter`）。

```python
import os
import argparse
import torch
import torch.distributed as dist


def init(backend: str):
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world = dist.get_world_size()
    return rank, world


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
    dist.reduce_scatter(out, list(big.chunk(world)), op=dist.ReduceOp.SUM)
    if rank == 0:
        print("[reduce_scatter] rank0 out", out.tolist())

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    # torchrun 会自动设置这些环境变量；手动启动则需要自己设
    for k in ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]:
        if k not in os.environ:
            pass
    main()
```

你可以先跑通 **AllReduce/AllGather** 的输出是否符合预期，再用 profiler 看耗时。

---

## 9. NCCL / 网络拓扑的常识（理解“为什么同样代码速度差很多”）

- **通信不只看带宽**：拓扑（NVLink vs PCIe）、跨 NUMA、跨机 IB、交换机 oversubscription 都会影响。
- **NCCL 选择算法**：会根据消息大小/拓扑选择 ring/tree/分层方案（不同版本策略不同）。
- **跨机训练**：通常机内（NVLink）快、机间（IB/RoCE）慢；分层算法（hierarchical allreduce）非常常见。

---

## 10. 常见坑与排查清单（你会立刻用到）

- **hang（卡死）**：
  - rank 数/进程数不一致、某些 rank 没有进入 collective。
  - 不同 rank 调用顺序不一致（A rank 先 allreduce，B rank 先 allgather）。
  - 多机下 master 地址/端口/防火墙问题。
- **性能差**：
  - 小张量太多：bucket 太小 / 没做融合。
  - 负载不均：AllToAll/MoE skew 导致尾部延迟。
  - 没有 overlap：通信在等计算或计算在等通信。

---

## 11. 练习（从“会用”到“会优化”）

- **练习 A（语义验证）**：把 `all_reduce(sum)` 换成 `MAX`/`AVG`（AVG 需要你自己除以 world_size），确认输出。
- **练习 B（消息大小 vs 性能）**：把张量长度从 1K、1M、100M 逐级增大，记录耗时，观察 α 与带宽主导区间。
- **练习 C（bucket 的影响）**：用 DDP 训练一个小模型，调 `bucket_cap_mb`，看 step time 变化。
- **练习 D（AllGather vs ReduceScatter 组合）**：实现一个“先 reduce_scatter 再 all_gather”的等价 allreduce，对比耗时。

---

## 12. 你接下来如果愿意，我可以按你的场景定制

为了把这份笔记变成“最对你胃口”的版本，你告诉我 3 个信息我就能继续补一版更贴近实战的：

- **你主要跑的硬件**：NVIDIA GPU（NVLink/PCIe）还是 Ascend/NPU？
- **你主要的并行方式**：DDP 为主？还是 TP/EP（MoE）？
- **你更关心**：语义入门、性能优化、还是排查 hang？

