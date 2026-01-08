# vLLM vs SGLang：CPU/GPU Overlap（异步流水化）实现对比笔记

这份笔记聚焦“**CPU/GPU overlap**”这一件事：如何把 **CPU 上的调度/后处理** 与 **GPU 上的 forward/采样/拷贝** 尽量重叠，以提高吞吐并降低 pipeline bubble。

> 你可以直接打开同目录的 `*.puml` 文件渲染图：它们把关键控制流、同步点、依赖关系画出来。

---

## 快速结论（先给你一个抓手）

- **vLLM 的 overlap 核心**：`EngineCore` 侧通过 **`non_block=True` 的 Future** + （PP 时）**`batch_queue`** 做“**先 schedule/提交下一批，再在合适时机取回上一批结果**”，从而减少等待、填满流水线。
- **SGLang 的 overlap 核心**：Scheduler 主循环 `event_loop_overlap()` 通过 **`result_queue`** 把“本轮 launch”与“上一轮结果处理”交错；`run_batch()` 使用 **独立 CUDA stream + non_blocking D2H copy + CUDA event** 把拷贝/CPU后处理隐藏在下一轮 forward 后面；并用 **`FutureMap`（负 token id 占位）** 显式表达跨 batch 依赖，让依赖解析也 GPU 化。

---

## 1) vLLM：CPU/GPU overlap 的实现要点（EngineCore + batch_queue）

### 1.1 基础路径：`step()`（异步提交但同步取回）
- `scheduler.schedule()`：CPU 生成本轮 `SchedulerOutput`
- `execute_model(..., non_block=True)`：提交 GPU 执行，返回 `Future`
- `future.result()`：同步等待结果（这条路径 overlap 能力有限，因为取回是阻塞的）

对应代码（你本地 repo）主要在：
- `vllm-main/vllm-main/vllm/v1/engine/core.py`

### 1.2 overlap 关键：`step_with_batch_queue()`（CPU 提前“填队列”）
当 `batch_queue_size > 1`（通常 PP 需要）时，`EngineCore` 会启用 `batch_queue`：
- **优先 schedule 并提交执行**，把 Future 入队
- 若队列未满且最老的 Future 还没 done，**直接返回**，继续下一轮 schedule（CPU 不阻塞）
- 只有在必须产出结果/队列满时才 `future.result()`，从队列尾部取“最早完成的批次”结果

这就是 vLLM 的“CPU/GPU overlap”主要来源：**把 blocking 点从“每一步”推迟到“必要的时候”**。

### 1.3 同步点来自哪里：grammar / structured output / spec decode 依赖
在 `step_with_batch_queue()` 中，如果某些请求存在 `pending_structured_output_tokens`：
- 需要先处理上一轮输出（例如 grammar 状态推进、draft token ids 更新）才能计算下一轮的 bitmask/采样
- 因而出现 `deferred_scheduler_output` 分支，形成“必须等”的同步点

这解释了：vLLM 的 overlap 并不是无条件的，它受“**跨步依赖**”约束。

---

## 2) SGLang：CPU/GPU overlap 的实现要点（event_loop_overlap + stream/event + FutureMap）

### 2.1 主循环层面：`event_loop_overlap()` 的交错执行
SGLang 的 loop 明确写成：
- 本轮：`recv -> schedule -> run_batch(batch)`（提交 GPU）
- 同时：从 `result_queue` 取“上一轮 batch_result”做 `process_batch_result(...)`（CPU 后处理）
- 再做一些必须依赖“上一轮处理完成”的事情（例如 `launch_batch_sample_if_needed` 有依赖说明）

这使得 CPU 的“结果处理/输出/释放/缓存更新”等逻辑，被系统性地放到 GPU forward 的空隙里。

### 2.2 GPU 层面：独立 stream + non_blocking D2H + event
在 `run_batch()` 的 overlap 分支里（generation case）：
- 在 `forward_stream_ctx` 中执行 `forward_batch_generation(...)`
- 拿到 `batch_result` 后，立刻做 `copy_to_cpu(..., non_blocking=True)` 并记录 `copy_done` event
- CPU 侧处理结果时会 `synchronize()`，保证读到的是已复制完成的数据

这让“GPU forward”和“GPU->CPU 拷贝 + CPU 后处理”可以更高概率重叠。

### 2.3 跨 batch 依赖：`FutureMap`（负 token id 占位 + GPU resolve）
SGLang 为了解决 overlap 下的“下一轮输入依赖上一轮 token”问题：
- 给每个 batch 分配 `future_indices`
- 把 `batch.output_ids` 写成 `-future_indices`（负数占位）
- 下一轮 forward 前 `future_map.resolve_future(...)`，用 `torch.where(input_ids<0, map[-id], id)` 把占位替换为真实 token

这个设计很关键：它把依赖解析做成一个**GPU 内的小算子**，避免 CPU 上的强同步。

### 2.4 overlap 的“禁用条件”：连续 prefill、spec+grammar 等
SGLang 在 `is_disable_overlap_for_batch()` 里显式写了 overlap 的禁用条件：
- 连续两个 prefill（为了改善第一批 TTFT）
- overlap + spec(v2) + grammar 目前不支持（需要 grammar_sync）

这等价于告诉你：SGLang overlap 也是在“依赖/正确性/性能权衡”下有选择地开启。

---

## 3) 对比总结：两者 overlap 的“设计风格”差异

### 3.1 overlap 的抓手不同
- **vLLM**：用 Future +（可选）queue 把 blocking 推迟；更像“任务系统/流水线填充”。
- **SGLang**：用 stream/event + D2H 异步拷贝 + 结果队列把 CPU/GPU 分工显式流水化；再用 FutureMap 解决跨步 token 依赖。

### 3.2 同步点来源不同
- **vLLM**：同步点多来自“采样/grammar/spec 需要上一轮结果”的控制依赖，以及必须取回 Future 的时刻。
- **SGLang**：同步点更多被显式控制在 `process_batch_result(...)` 里（例如 `copy_done.synchronize()`），并且会在 `is_disable_overlap_for_batch()` 主动降级。

---

## 4) PlantUML 图（请打开这些文件渲染）

- `vllm_cpu_gpu_overlap_batch_queue.puml`：vLLM batch_queue 如何把 schedule/execute 与等待结果交错
- `vllm_cpu_gpu_overlap_dependency_points.puml`：vLLM 在 structured output / spec decode 下的同步点
- `sglang_cpu_gpu_overlap_event_loop.puml`：SGLang event_loop_overlap 的交错控制流
- `sglang_futuremap_overlap.puml`：SGLang FutureMap（负 token id）如何解析跨 batch 依赖
- `compare_cpu_gpu_overlap.puml`：同一张图并列对比两者的 overlap 手段、同步点与可重叠段

## 5) 三方对比（新增：MindIE-LLM）

- 综述文档：`async_scheduling_three_way_compare.md`
- `mindie_async_plugin_manager_pipeline.puml`：MindIE Python PluginManager 的 async_infer 流水线
- `mindie_scheduler_placeholder_async.puml`：MindIE C++ Scheduler 的 placeholder token 异步推进
- `three_way_async_scheduling_compare.puml`：vLLM vs SGLang vs MindIE-LLM 的总对比图
- `mindie_pipeline_refactor_compare.puml`：MindIE 当前“双层异步”与“单一主流水”收敛建议对比图

