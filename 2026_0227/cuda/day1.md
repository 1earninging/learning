## Day1：CUDA 编程模型 + 索引 + warp/SIMT（配套最小工程）

对应计划：`2026_0215/CUDA_Triton_面试准备_45天.md` Day 1

### 今日交付（对齐原计划）
- **写**：CUDA kernel 做 vector add（含边界检查），验证正确性
- **讲**：warp 是什么？为什么 divergence 会慢？
- **测**：记录 kernel time；对比不同 blockDim（128/256/512）

---

### 1) CUDA 编程模型 30 秒复述
- **grid**：一整个 kernel launch 的线程集合
- **block**：grid 的分块（线程块），块内线程可用 shared memory、可同步（`__syncthreads()`）
- **thread**：执行同一段 kernel 代码的最小执行单元（从编程视角）

索引计算（最常用 1D）：
```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

边界检查（必须要有，否则越界就是 UB/非法访存）：
```cpp
if (idx < n) { ... }
```

---

### 2) warp / SIMT 是什么（2 分钟口述稿）
- **warp**：硬件调度的基本单位；在 NVIDIA GPU 上通常是 **32 个线程**一组。
- **SIMT**（Single Instruction, Multiple Threads）：同一 warp 的线程在同一时刻执行**同一条指令**，但每个线程有自己的寄存器/数据。

一句话：**你写的是 thread 级代码，但硬件按 warp 打包执行。**

---

### 3) 为什么 divergence（分支发散）会慢
当同一个 warp 里的线程遇到 `if/else`，并且“有的线程走 if、有的线程走 else”：
- 硬件通常会把两条路径都执行一遍（用掩码 mask 让不该执行的线程空转）
- 于是 **有效并行度下降**，吞吐变差

典型例子（同 warp 内一半线程走 true，一半走 false）：
```cpp
if (idx % 2 == 0) {
  // path A
} else {
  // path B
}
```

经验判断：
- **warp 内分歧越随机** → 越难优化 → 越慢
- 如果分支是“整 warp 同时同向”（例如按 tile/块对齐的条件），影响会小很多

---

### 4) 实战：vector add（写 + 测 + 验证）
代码：`2026_0227/cuda/day1_vector_add.cu`

你会看到它做了三件事：
- **正确性**：CPU 端计算参考值，对比 GPU 输出（误差阈值）
- **计时**：CUDA events 计 kernel 平均耗时（warmup + 多次迭代取均值）
- **对比 blockDim**：128 / 256 / 512

编译运行（本机需要 CUDA Toolkit / nvcc）：
```bash
cd /mnt/d/github/learning/2026_0227/cuda
nvcc -O3 -std=c++17 day1_vector_add.cu -o day1_vector_add
./day1_vector_add
./day1_vector_add 10000000
```

怎么看输出：
- `time=... ms`：kernel 平均耗时
- `effective_bw=... GB/s`：按“读 a+b、写 c”的字节数估算的有效带宽（粗略但直观）

---

### 5) blockDim 选 128/256/512 的直觉
- blockDim 太小：每个 block 线程少，可能不足以隐藏延迟
- blockDim 太大：寄存器/资源占用可能更高，影响并发 block 数（occupancy）
- 对这种 **memory-bound** 的简单 kernel，最佳点通常跟 GPU 架构、编译器、内存系统有关，所以用“测”来定是最稳的。

