### 45 天面试准备计划（CUDA + Triton｜Kernel/性能专项）
**起始日期**：2026-02-15（Day 1）
**定位**：面向“会写 kernel + 会调性能 + 会解释 GPU 原理”的面试。覆盖 CUDA 编程模型、性能分析、常见并行算法与 Triton kernel 开发/调优。
**建议时间**：每天 2–3 小时（学习/口述 45min + 编码 60–90min + profiling/复盘 30min）。

---
### 每天固定流程（面试版）
- **口述 10min**：用 2 分钟讲清楚昨天的 1 个核心概念（例如 coalescing/occupancy/warp divergence）。
- **编码 60–90min**：写 1 个最小可运行 kernel（CUDA 或 Triton），要求 correctness + 基础性能。
- **Profiling 30min**：用 Nsight Systems/Compute（或 triton benchmark）验证 3 个指标：吞吐/带宽/占用率或 kernel time。
- **复盘 10min**：记录 3 行：瓶颈在哪里→怎么改→改完指标怎么变。

---
### 间隔复习规则（GPU/Kernel 知识点）
- **D+1**：闭卷画图（线程块/warp/内存层级）+ 口述 2 分钟
- **D+4**：闭卷重写最小 kernel（只写核心循环/索引/边界）
- **D+10**：换 tile/block 参数重调一次，并解释为什么快/慢
- **D+21**：模拟面试：给一个 kernel/图表，要求你定位瓶颈并提出改法

---
### 阶段总览
- **第 1 周**：CUDA 基础模型 + 内存层级 + 索引正确性
- **第 2 周**：性能关键：访存合并、shared memory、bank conflict、occupancy
- **第 3 周**：并行算法组件：reduction/scan/softmax/gemm 直觉 + 原子/同步
- **第 4 周**：Triton 基础：program_id/tl.load/store、block pointers、autotune
- **第 5 周**：Triton 高级：pipelining(num_stages)、layout、mask、fp16/bf16/fp8
- **第 6 周**：综合：手写 Attention 子模块（QK/softmax/V）、paged/indexed、模拟面试

---
### Day-by-Day（45 天）
#### Day 1（2026-02-15）
- **主题**：CUDA 编程模型：grid/block/thread、warp、SIMT，索引计算
- **当日交付（写出来/讲出来/测出来）**：
  - 写：CUDA kernel 做 vector add（含边界检查），验证正确性。
  - 讲：warp 是什么？为什么 divergence 会慢？
  - 测：记录 kernel time；对比不同 blockDim（128/256/512）。

#### Day 2（2026-02-16）
- **主题**：CUDA 内存层级：global/L2/shared/register、延迟与带宽直觉
- **当日交付（写出来/讲出来/测出来）**：
  - 写：CUDA kernel 做 memcpy（read-only / write-only），做简单带宽测试。
  - 讲：带宽上限怎么估？roofline 思路（概念）。
- **复习（D-1）**：回忆 Day 1 的 1 个概念 + 1 个优化点，2 分钟讲清楚。

#### Day 3（2026-02-17）
- **主题**：访存合并（coalescing）与对齐；AoS vs SoA
- **当日交付（写出来/讲出来/测出来）**：
  - 写：用两种布局读写（AoS/SoA）并对比带宽。
  - 讲：什么访问模式会产生多次 memory transaction？
- **复习（D-1）**：回忆 Day 2 的 1 个概念 + 1 个优化点，2 分钟讲清楚。

#### Day 4（2026-02-18）
- **主题**：shared memory 基础：tile、同步 __syncthreads
- **当日交付（写出来/讲出来/测出来）**：
  - 写：naive matmul（小尺寸）+ shared tiled matmul（正确性优先）。
  - 讲：为什么 shared 能提速？什么时候反而慢？
- **复习（D-1）**：回忆 Day 3 的 1 个概念 + 1 个优化点，2 分钟讲清楚。

#### Day 5（2026-02-19）
- **主题**：shared memory bank conflict 与 padding
- **当日交付（写出来/讲出来/测出来）**：
  - 写：一个 shared transpose kernel（naive vs padding 避免 bank conflict）。
  - 测：用 Nsight Compute 看 shared_load/store conflict 指标（或对比时间）。
- **复习（D-1）**：回忆 Day 4 的 1 个概念 + 1 个优化点，2 分钟讲清楚。
- **复习（D-4）**：闭卷重写 Day 1 的最小 kernel（索引+mask+边界）。

#### Day 6（2026-02-20）
- **主题**：occupancy：寄存器/共享内存/线程数限制；block size 选择
- **当日交付（写出来/讲出来/测出来）**：
  - 写：让同一 kernel 在不同 block size 下跑，记录 occupancy/时间（概念+实践）。
  - 讲：occupancy 高一定快吗？举反例（memory-bound）。
- **复习（D-1）**：回忆 Day 5 的 1 个概念 + 1 个优化点，2 分钟讲清楚。
- **复习（D-4）**：闭卷重写 Day 2 的最小 kernel（索引+mask+边界）。

#### Day 7（2026-02-21）
- **主题**：周复盘：把本周 10 个概念写成 2 分钟口述稿
- **当日交付（写出来/讲出来/测出来）**：
  - 模拟面试：给你一个慢 kernel（访存不合并/分支多），你如何定位？
  - 整理：索引/边界/对齐的常见 bug 清单。
- **复习（D-1）**：回忆 Day 6 的 1 个概念 + 1 个优化点，2 分钟讲清楚。
- **复习（D-4）**：闭卷重写 Day 3 的最小 kernel（索引+mask+边界）。

#### Day 8（2026-02-22）
- **主题**：并行 reduction：tree reduction、warp shuffle 入门
- **当日交付（写出来/讲出来/测出来）**：
  - 写：sum reduction（naive atomic vs block reduction）。
  - 讲：为什么 atomic 慢？如何减少原子次数？
- **复习（D-1）**：回忆 Day 7 的 1 个概念 + 1 个优化点，2 分钟讲清楚。
- **复习（D-4）**：闭卷重写 Day 4 的最小 kernel（索引+mask+边界）。

#### Day 9（2026-02-23）
- **主题**：warp primitives：__shfl_*、warp-level reduction
- **当日交付（写出来/讲出来/测出来）**：
  - 写：warp reduce（shuffle）并与 shared reduce 对比。
  - 讲：warp 同步与 __syncthreads 的区别。
- **复习（D-1）**：回忆 Day 8 的 1 个概念 + 1 个优化点，2 分钟讲清楚。
- **复习（D-4）**：闭卷重写 Day 5 的最小 kernel（索引+mask+边界）。

#### Day 10（2026-02-24）
- **主题**：原子操作与一致性：atomicAdd、CAS，热点冲突
- **当日交付（写出来/讲出来/测出来）**：
  - 写：histogram（小桶数）并观察热点冲突；提出分桶/分块优化。
  - 讲：什么时候用 block-private histogram？
- **复习（D-1）**：回忆 Day 9 的 1 个概念 + 1 个优化点，2 分钟讲清楚。
- **复习（D-4）**：闭卷重写 Day 6 的最小 kernel（索引+mask+边界）。

#### Day 11（2026-02-25）
- **主题**：前缀和 scan（概念）：Blelloch scan，分块 + 递归
- **当日交付（写出来/讲出来/测出来）**：
  - 写：先做 CPU 版 scan，再写一个 block 内 scan（小规模）CUDA kernel。
  - 讲：scan 和 reduction 的关系。
- **复习（D-1）**：回忆 Day 10 的 1 个概念 + 1 个优化点，2 分钟讲清楚。
- **复习（D-4）**：闭卷重写 Day 7 的最小 kernel（索引+mask+边界）。

#### Day 12（2026-02-26）
- **主题**：异步与流水：streams、cudaMemcpyAsync、overlap
- **当日交付（写出来/讲出来/测出来）**：
  - 写：两 stream overlap（H2D copy + kernel）示例（可用 pinned memory）。
  - 讲：什么条件才能 overlap？
- **复习（D-1）**：回忆 Day 11 的 1 个概念 + 1 个优化点，2 分钟讲清楚。
- **复习（D-4）**：闭卷重写 Day 8 的最小 kernel（索引+mask+边界）。

#### Day 13（2026-02-27）
- **主题**：并发执行与 launch overhead：kernel fusion 直觉
- **当日交付（写出来/讲出来/测出来）**：
  - 写：两段 kernel（A->B）与 fused kernel，对比时间（哪怕小提升）。
  - 讲：为什么 launch overhead 在小 batch 上更显著？
- **复习（D-1）**：回忆 Day 12 的 1 个概念 + 1 个优化点，2 分钟讲清楚。
- **复习（D-4）**：闭卷重写 Day 9 的最小 kernel（索引+mask+边界）。

#### Day 14（2026-02-28）
- **主题**：周复盘：性能三件套：带宽/算力/占用率如何判别瓶颈
- **当日交付（写出来/讲出来/测出来）**：
  - 模拟面试：给 roofline 数据，判断是 memory-bound 还是 compute-bound。
- **复习（D-1）**：回忆 Day 13 的 1 个概念 + 1 个优化点，2 分钟讲清楚。
- **复习（D-4）**：闭卷重写 Day 10 的最小 kernel（索引+mask+边界）。

#### Day 15（2026-03-01）
- **主题**：softmax 数值稳定：max-subtraction、log-sum-exp
- **当日交付（写出来/讲出来/测出来）**：
  - 写：CUDA softmax（对每行），实现稳定版；对比不稳定版。
  - 讲：为什么要减 max？溢出在哪一步发生？
- **复习（D-1）**：回忆 Day 14 的 1 个概念 + 1 个优化点，2 分钟讲清楚。
- **复习（D-4）**：闭卷重写 Day 11 的最小 kernel（索引+mask+边界）。

#### Day 16（2026-03-02）
- **主题**：layernorm/rmsnorm：均值方差/均方根，融合思路
- **当日交付（写出来/讲出来/测出来）**：
  - 写：CUDA layernorm（每行），再尝试 fused（读一次写一次）。
  - 讲：Welford 算法（概念即可）。
- **复习（D-1）**：回忆 Day 15 的 1 个概念 + 1 个优化点，2 分钟讲清楚。
- **复习（D-4）**：闭卷重写 Day 12 的最小 kernel（索引+mask+边界）。

#### Day 17（2026-03-03）
- **主题**：GEMM 直觉：tile、寄存器 blocking、vectorized load
- **当日交付（写出来/讲出来/测出来）**：
  - 写：小 GEMM（fp16 输入，fp32 accumulate），解释为什么要 accumulate 到 fp32。
  - 讲：Tensor Core 条件（概念：对齐/数据类型/矩阵尺寸）。
- **复习（D-1）**：回忆 Day 16 的 1 个概念 + 1 个优化点，2 分钟讲清楚。
- **复习（D-4）**：闭卷重写 Day 13 的最小 kernel（索引+mask+边界）。

#### Day 18（2026-03-04）
- **主题**：attention 组成：QK^T、mask、softmax、PV
- **当日交付（写出来/讲出来/测出来）**：
  - 写：先实现一个简化 attention（小尺寸，正确性）。
  - 讲：FlashAttention 的核心优化点是什么（不物化 QK/流式 softmax）。
- **复习（D-1）**：回忆 Day 17 的 1 个概念 + 1 个优化点，2 分钟讲清楚。
- **复习（D-4）**：闭卷重写 Day 14 的最小 kernel（索引+mask+边界）。

#### Day 19（2026-03-05）
- **主题**：paged/indexed KV cache 直觉：indptr/indices gather
- **当日交付（写出来/讲出来/测出来）**：
  - 写：用 indices gather K/V 的 attention（参考 CSR），正确性优先。
  - 讲：为什么 paged 会破坏连续访存？如何用分块/split-K 缓解？
- **复习（D-1）**：回忆 Day 18 的 1 个概念 + 1 个优化点，2 分钟讲清楚。
- **复习（D-4）**：闭卷重写 Day 15 的最小 kernel（索引+mask+边界）。

#### Day 20（2026-03-06）
- **主题**：调试与正确性：race、越界、未初始化；cuda-memcheck 思路
- **当日交付（写出来/讲出来/测出来）**：
  - 写：故意制造一个越界 bug，再定位修复（记录过程）。
  - 讲：如何写 kernel 的单元测试（随机输入+对比 torch/CPU）。
- **复习（D-1）**：回忆 Day 19 的 1 个概念 + 1 个优化点，2 分钟讲清楚。
- **复习（D-4）**：闭卷重写 Day 16 的最小 kernel（索引+mask+边界）。

#### Day 21（2026-03-07）
- **主题**：周复盘：把 reduction/softmax/attention 画成算子图
- **当日交付（写出来/讲出来/测出来）**：
  - 模拟面试：现场讲清楚一个 attention kernel 的内存访问与瓶颈。
- **复习（D-1）**：回忆 Day 20 的 1 个概念 + 1 个优化点，2 分钟讲清楚。
- **复习（D-4）**：闭卷重写 Day 17 的最小 kernel（索引+mask+边界）。

#### Day 22（2026-03-08）
- **主题**：Triton 编程模型：program_id、tl.arange、mask，指针算术
- **当日交付（写出来/讲出来/测出来）**：
  - 写：Triton vector add（含 mask），对比 torch。
  - 讲：为什么 Triton 里“block”是编译期常量（tl.constexpr）很重要？
- **复习（D-1）**：回忆 Day 21 的 1 个概念 + 1 个优化点，2 分钟讲清楚。
- **复习（D-4）**：闭卷重写 Day 18 的最小 kernel（索引+mask+边界）。

#### Day 23（2026-03-09）
- **主题**：Triton tl.load/tl.store、stride、layout（行主/列主）
- **当日交付（写出来/讲出来/测出来）**：
  - 写：Triton 2D copy/transpose（含 mask）。
  - 讲：stride 与 view/reshape 的区别（以及如何导致错误）。
- **复习（D-1）**：回忆 Day 22 的 1 个概念 + 1 个优化点，2 分钟讲清楚。
- **复习（D-4）**：闭卷重写 Day 19 的最小 kernel（索引+mask+边界）。

#### Day 24（2026-03-10）
- **主题**：Triton matmul 入门：BLOCK_M/N/K、num_warps
- **当日交付（写出来/讲出来/测出来）**：
  - 写：Triton matmul（fp16），做简单基准（triton.testing.do_bench）。
  - 讲：num_warps 怎么影响并行度与寄存器压力？
- **复习（D-1）**：回忆 Day 23 的 1 个概念 + 1 个优化点，2 分钟讲清楚。
- **复习（D-4）**：闭卷重写 Day 20 的最小 kernel（索引+mask+边界）。

#### Day 25（2026-03-11）
- **主题**：Triton reduction/softmax：稳定 softmax 的写法
- **当日交付（写出来/讲出来/测出来）**：
  - 写：Triton softmax（row-wise），对比 torch.softmax。
  - 讲：mask 在 softmax 里怎么处理（-inf）。
- **复习（D-1）**：回忆 Day 24 的 1 个概念 + 1 个优化点，2 分钟讲清楚。
- **复习（D-4）**：闭卷重写 Day 21 的最小 kernel（索引+mask+边界）。

#### Day 26（2026-03-12）
- **主题**：Triton autotune：key、configs、启发式
- **当日交付（写出来/讲出来/测出来）**：
  - 写：给 matmul 或 softmax 加 autotune（2–3 组 config）。
  - 讲：为什么 autotune 的 key 选择很关键？
- **复习（D-1）**：回忆 Day 25 的 1 个概念 + 1 个优化点，2 分钟讲清楚。
- **复习（D-4）**：闭卷重写 Day 22 的最小 kernel（索引+mask+边界）。

#### Day 27（2026-03-13）
- **主题**：Triton 与 torch.compile/动态图：限制与实践
- **当日交付（写出来/讲出来/测出来）**：
  - 讲：Triton kernel 什么时候会触发 recompilation？如何减少？
  - 写：把 Triton kernel 封装成 python API，写最小测试。
- **复习（D-1）**：回忆 Day 26 的 1 个概念 + 1 个优化点，2 分钟讲清楚。
- **复习（D-4）**：闭卷重写 Day 23 的最小 kernel（索引+mask+边界）。

#### Day 28（2026-03-14）
- **主题**：周复盘：Triton 基础套路总结（索引/stride/mask/constexpr）
- **当日交付（写出来/讲出来/测出来）**：
  - 模拟面试：给一段 Triton kernel，让你找 bug（stride/mask 错）。
- **复习（D-1）**：回忆 Day 27 的 1 个概念 + 1 个优化点，2 分钟讲清楚。
- **复习（D-4）**：闭卷重写 Day 24 的最小 kernel（索引+mask+边界）。

#### Day 29（2026-03-15）
- **主题**：Triton pipelining：num_stages、软件流水直觉
- **当日交付（写出来/讲出来/测出来）**：
  - 写：调整 num_stages 比较性能（固定输入形状）。
  - 讲：为什么更深流水不一定更快？（寄存器/占用率）
- **复习（D-1）**：回忆 Day 28 的 1 个概念 + 1 个优化点，2 分钟讲清楚。
- **复习（D-4）**：闭卷重写 Day 25 的最小 kernel（索引+mask+边界）。

#### Day 30（2026-03-16）
- **主题**：向量化与对齐：tl.multiple_of/tl.max_contiguous
- **当日交付（写出来/讲出来/测出来）**：
  - 写：在 kernel 中加入 multiple_of/max_contiguous hint 并观测变化。
  - 讲：对齐为什么影响内存事务与吞吐？
- **复习（D-1）**：回忆 Day 29 的 1 个概念 + 1 个优化点，2 分钟讲清楚。
- **复习（D-4）**：闭卷重写 Day 26 的最小 kernel（索引+mask+边界）。

#### Day 31（2026-03-17）
- **主题**：FP16/BF16/FP8 直觉：scale、accumulate、误差
- **当日交付（写出来/讲出来/测出来）**：
  - 讲：FP8 为什么需要 scale？KV cache FP8 的典型处理流程。
  - 写：实现一个带 scale 的量化/反量化小函数（CPU 或 Triton）。
- **复习（D-1）**：回忆 Day 30 的 1 个概念 + 1 个优化点，2 分钟讲清楚。
- **复习（D-4）**：闭卷重写 Day 27 的最小 kernel（索引+mask+边界）。

#### Day 32（2026-03-18）
- **主题**：split-K / 分块并行与归并（stage1/stage2）
- **当日交付（写出来/讲出来/测出来）**：
  - 写：模仿 decode_attention 的两阶段：stage1 写 partial + lse，stage2 合并。
  - 讲：为什么要两阶段？如何保证数值稳定？
- **复习（D-1）**：回忆 Day 31 的 1 个概念 + 1 个优化点，2 分钟讲清楚。
- **复习（D-4）**：闭卷重写 Day 28 的最小 kernel（索引+mask+边界）。

#### Day 33（2026-03-19）
- **主题**：Triton 注意力：prefill vs decode 的差异
- **当日交付（写出来/讲出来/测出来）**：
  - 讲：prefill 是矩阵乘/大 tile；decode 是单 query + 大 KV gather。
  - 写：实现一个简化 decode attention（只做 causal+无窗口）。
- **复习（D-1）**：回忆 Day 32 的 1 个概念 + 1 个优化点，2 分钟讲清楚。
- **复习（D-4）**：闭卷重写 Day 29 的最小 kernel（索引+mask+边界）。

#### Day 34（2026-03-20）
- **主题**：工程化：kernel 参数、单测、benchmark、回归
- **当日交付（写出来/讲出来/测出来）**：
  - 写：给你的 kernel 写 unittest（随机种子+误差阈值）。
  - 写：写一个 benchmark 脚本输出 p50/p99（最简）。
- **复习（D-1）**：回忆 Day 33 的 1 个概念 + 1 个优化点，2 分钟讲清楚。
- **复习（D-4）**：闭卷重写 Day 30 的最小 kernel（索引+mask+边界）。

#### Day 35（2026-03-21）
- **主题**：周复盘：把你写过的 Triton kernel 整理成模板库
- **当日交付（写出来/讲出来/测出来）**：
  - 模拟面试：给目标（加速 softmax/attention），你如何设计算法与指标？
- **复习（D-1）**：回忆 Day 34 的 1 个概念 + 1 个优化点，2 分钟讲清楚。
- **复习（D-4）**：闭卷重写 Day 31 的最小 kernel（索引+mask+边界）。

#### Day 36（2026-03-22）
- **主题**：综合 1：实现一个“带 mask 的 stable softmax”并优化
- **当日交付（写出来/讲出来/测出来）**：
  - 写：CUDA 或 Triton 任选，做 2 轮优化（减少读写/更好 tile）。
  - 讲：你优化前后的瓶颈对比。
- **复习（D-1）**：回忆 Day 35 的 1 个概念 + 1 个优化点，2 分钟讲清楚。
- **复习（D-4）**：闭卷重写 Day 32 的最小 kernel（索引+mask+边界）。

#### Day 37（2026-03-23）
- **主题**：综合 2：实现一个简单 flash-style attention（流式 softmax）
- **当日交付（写出来/讲出来/测出来）**：
  - 写：不要求极致性能，但要能解释为什么不需要存 QK。
  - 讲：和普通 attention 的内存差别。
- **复习（D-1）**：回忆 Day 36 的 1 个概念 + 1 个优化点，2 分钟讲清楚。
- **复习（D-4）**：闭卷重写 Day 33 的最小 kernel（索引+mask+边界）。

#### Day 38（2026-03-24）
- **主题**：综合 3：实现 paged/indexed decode attention（indptr+indices）
- **当日交付（写出来/讲出来/测出来）**：
  - 写：支持 kv_splits（split-K），输出中间 lse，再合并。
  - 讲：为什么它更像服务端引擎里的 kernel？
- **复习（D-1）**：回忆 Day 37 的 1 个概念 + 1 个优化点，2 分钟讲清楚。
- **复习（D-4）**：闭卷重写 Day 34 的最小 kernel（索引+mask+边界）。

#### Day 39（2026-03-25）
- **主题**：综合 4：KV cache 写入/reshape（slot_mapping 思路）
- **当日交付（写出来/讲出来/测出来）**：
  - 讲：slot_mapping 是什么？block_idx/block_offset 如何定位 cache。
  - 写：实现一个简化 reshape_and_cache（Triton）。
- **复习（D-1）**：回忆 Day 38 的 1 个概念 + 1 个优化点，2 分钟讲清楚。
- **复习（D-4）**：闭卷重写 Day 35 的最小 kernel（索引+mask+边界）。

#### Day 40（2026-03-26）
- **主题**：性能面试：看 Nsight 图/表定位瓶颈
- **当日交付（写出来/讲出来/测出来）**：
  - 练：准备 6 张你自己的 profiling 截图（或文字记录），每张 2 分钟讲清楚。
- **复习（D-1）**：回忆 Day 39 的 1 个概念 + 1 个优化点，2 分钟讲清楚。
- **复习（D-4）**：闭卷重写 Day 36 的最小 kernel（索引+mask+边界）。

#### Day 41（2026-03-27）
- **主题**：查漏补缺：把最不熟的 6 个点补齐
- **当日交付（写出来/讲出来/测出来）**：
  - 补：bank conflict / atomic / occupancy / autotune / split-K / fp8 任选四。
- **复习（D-1）**：回忆 Day 40 的 1 个概念 + 1 个优化点，2 分钟讲清楚。
- **复习（D-4）**：闭卷重写 Day 37 的最小 kernel（索引+mask+边界）。

#### Day 42（2026-03-28）
- **主题**：系统化整理：写 1 页“GPU Kernel 面试速记”
- **当日交付（写出来/讲出来/测出来）**：
  - 输出：coalescing、shared、occupancy、warp、softmax 稳定、split-K、indices gather。
- **复习（D-1）**：回忆 Day 41 的 1 个概念 + 1 个优化点，2 分钟讲清楚。
- **复习（D-4）**：闭卷重写 Day 38 的最小 kernel（索引+mask+边界）。

#### Day 43（2026-03-29）
- **主题**：终局模拟面试 1：CUDA kernel 现场手写 + 口述优化
- **当日交付（写出来/讲出来/测出来）**：
  - 题型：reduction/softmax/matmul 三选一，要求正确性+性能思路。
- **复习（D-1）**：回忆 Day 42 的 1 个概念 + 1 个优化点，2 分钟讲清楚。
- **复习（D-4）**：闭卷重写 Day 39 的最小 kernel（索引+mask+边界）。

#### Day 44（2026-03-30）
- **主题**：终局模拟面试 2：Triton kernel 现场手写 + autotune 方案
- **当日交付（写出来/讲出来/测出来）**：
  - 题型：softmax 或 indexed gather + reduce。
- **复习（D-1）**：回忆 Day 43 的 1 个概念 + 1 个优化点，2 分钟讲清楚。
- **复习（D-4）**：闭卷重写 Day 40 的最小 kernel（索引+mask+边界）。

#### Day 45（2026-03-31）
- **主题**：收官：整理个人作品集（kernel 列表 + benchmark 数据 + 经验总结）
- **当日交付（写出来/讲出来/测出来）**：
  - 输出：README（你写过哪些 kernel、各自指标、踩坑与修复）。
- **复习（D-1）**：回忆 Day 44 的 1 个概念 + 1 个优化点，2 分钟讲清楚。
- **复习（D-4）**：闭卷重写 Day 41 的最小 kernel（索引+mask+边界）。

---
### 高频面试问答清单（建议背到能脱口而出）
#### CUDA / GPU
- warp 与 thread 的关系？warp size 固定吗？
- 什么是 coalesced memory access？举例说明不合并会发生什么。
- shared memory 的作用？bank conflict 如何产生与规避？
- occupancy 是什么？高 occupancy 一定快吗？
- 寄存器溢出（spill）会带来什么后果？
- atomic 的热点冲突如何优化？
- 如何判断 kernel 是 memory-bound 还是 compute-bound？
- streams 如何 overlap copy/compute？需要什么条件？
- 为什么 data race 在 C/CUDA 里是 UB（或不确定行为）？怎么避免？

#### Triton
- Triton 的 program_id 对应什么？它和 CUDA block/thread 的类比是什么？
- tl.constexpr 为什么重要？哪些参数必须 constexpr？
- mask 的正确使用姿势是什么？softmax 里的 mask 怎么处理？
- num_warps/num_stages 各自影响什么？如何选？
- autotune 的 key 怎么设计，如何避免过度编译？
- 如何处理 stride/layout，避免 view/reshape 导致的错读？

#### Attention/Serving 相关
- FlashAttention 的核心思路是什么？如何做到不物化 QK^T？
- decode attention 为什么常用 split-K 两阶段？怎么合并 LSE 保证稳定？
- paged/indexed KV cache（indptr/indices）为什么常见？它的性能挑战是什么？
