# vLLM / SGLang / MindIE-LLM：异步调度实现对比（含 CPU/GPU overlap）

本文基于你当前工作区的源码阅读，对 **vLLM v1**、**SGLang SRT**、**MindIE-LLM** 的“异步调度/流水化执行”做一个三方对比。重点不讲概念，直接落到：

- **异步的切分边界**：哪些阶段可以并行，哪些必须同步
- **队列/线程/进程模型**：谁推进主循环，怎么避免阻塞
- **跨 step 依赖的表达**：下一步输入依赖上一步 token/grammar/spec 时怎么处理

---

## 0) 一句话抓手

- **vLLM**：通过 **Future +（PP 时）batch_queue** 把阻塞点后移；异步调度在 v1 里也通过“placeholder / output_placeholders”等机制消化跨步依赖。
- **SGLang**：通过 **event_loop_overlap + forward_stream/event + result_queue + FutureMap** 把 CPU 后处理与 GPU 前向系统性流水化，且把“跨 batch token 依赖”编码成负 token id 占位并在 GPU 上 resolve。
- **MindIE-LLM**：异步分两层：
  - **C++ Scheduler**：用 **PLACEHOLDER_TOKEN（-1）** + `predictedTokensBySeqId_` 显式维护“未回来的 token”，并用 `PrepareNextSchedule()` 在入队前**预累计 computed tokens**、**预插入 placeholder**（允许 outstanding 多个 batch）。
  - **Python PluginManager**：在 `async_infer` 模式下用 **forward 线程 + input/output 双队列 + Event 同步点** 做“上一轮结果后处理 + 下一轮 forward 提交”的 pipeline（非常像 SGLang 的 result_queue，但实现载体是 Python 线程/队列）。

---

## 1) MindIE-LLM：异步调度的两个落点

### 1.1 C++ Scheduler：placeholder token + outstanding batch 上限
关键代码在：
- `MindIE-LLM/src/scheduler/scheduler.cpp`
- `MindIE-LLM/src/scheduler/scheduler.h`

核心机制：
- `activateAsyncInference` 打开后，`maxScheduledBatch_` 会变大（与 `MAX_ASYNC_SCHEDULE_TIMES` 或 layerwise 参数有关）
- 每轮调度后，`PrepareNextSchedule()` 会：
  - `AccumulateComputedTokens(...)`：**提前把 num_computed_tokens 往前推**（避免下一轮重复算 KV）
  - `AddNextTokenPlaceHolder(...)`：在 `outputTokenIds` 末尾插入 **PLACEHOLDER_TOKEN**
- 执行器/响应线程把生成的 token 通过 `FetchSeqGeneratedTokens(...)` 注入 `predictedTokensBySeqId_`
- 调度器再用 `ReplacePlaceHolderWithToken(...)` 把尾部 placeholder 替换成真实 token

特别点：
- placeholder 数量会被严格限制（考虑 speculation/MTP：每轮 token 预算为 \(1+\gamma\)），避免长时间不命中/不回包导致 KV 空间被 placeholder 占满。
- chunked prefill 的非最后一块会跳过 placeholder（避免错误对齐）。

直观理解：MindIE 在 C++ 调度层把“token 回来之前”当作一种**显式状态**，靠 placeholder 把流水线推进下去。

### 1.2 Python PluginManager：双队列 + forward 线程流水
关键代码在：
- `MindIE-LLM/mindie_llm/text_generator/plugins/plugin_manager.py`

当 `async_infer=True`：
- 创建 `input_queue` / `output_queue`
- 启动 `forward_thread` 跑 `forward_loop()`

`generate_token_async()` 的结构非常像“流水线两阶段”：
1) **本轮预处理**：准备 `model_input_wrapper`
2) **取回上一轮结果**：`model_output_wrapper = output_queue.get()`
3) 用上一轮的 `sampling_output`/mask 去“补全本轮输入”（`_fill_in_model_result`），然后 `synchronize()`
4) 把本轮输入丢到 `input_queue`，让 forward 线程开始算“本轮”
5) 同时在主线程对“上一轮结果”做 `postprocess` 并输出

同步点：
- `launch_done`：用于保证“下一轮 forward 已经 launch”再去 postprocess（避免过度堆积/数据竞争）
- `postprocess_done`：forward 线程在 verify 前会等主线程 postprocess 完成（保证插件校验/缓存更新时序正确）

直观理解：MindIE 在 Python 层实现了一个“**prev_result 后处理**”与“**next_forward 提交**”交错的 pipeline；在 C++ 层又实现了更底层的 placeholder 异步推进。

---

## 2) vLLM：异步调度的抓手（Future + batch_queue + async placeholder 语义）

关键落点：
- `vllm-main/vllm-main/vllm/v1/engine/core.py`：`step_with_batch_queue()`（PP/多批并发）
- `vllm-main/vllm-main/vllm/v1/core/sched/async_scheduler.py`：异步调度下对 request 状态/placeholder 的修正逻辑

vLLM 的异步风格更“抽象化”：
- 执行层：`execute_model(..., non_block=True)` 返回 Future；batch_queue 让 CPU 尽量先 schedule/submit
- 调度层：不直接把 placeholder token 写进 token_ids list，而是通过 `num_output_placeholders` 等状态来表达“已调度但还未真正生成/落盘”的 token

---

## 3) SGLang：异步调度的抓手（显式 overlap loop + FutureMap）

关键落点：
- `sglang-main/.../scheduler.py`：`event_loop_overlap()` + `run_batch()`（stream/event + result_queue）
- `sglang-main/.../overlap_utils.py`：`FutureMap`（负 token id 占位 + GPU resolve）

SGLang 的异步更“系统工程化”：
- loop 内先 launch 当前 batch，再处理上一批结果（result_queue）
- GPU 上用 stream/event 把 D2H copy 与 CPU 处理重叠
- 用 FutureMap 解决“下一轮输入依赖上一轮 token”的依赖解析，尽量避免 CPU 等待

---

## 4) 三方对比表（高密度）

| 维度 | vLLM | SGLang | MindIE-LLM |
|---|---|---|---|
| **主循环载体** | EngineCore busy loop（多进程 + IO 线程/async client） | Python scheduler loop（多套 event_loop_*） | C++ Scheduler（调度策略/队列） + Python PluginManager（可选 async pipeline） |
| **异步的核心结构** | Future + batch_queue（填 pipeline） | result_queue + forward_stream/event + FutureMap | C++：placeholder token + predictedTokens map；Python：input/output queue + forward thread |
| **跨步 token 依赖表达** | request 状态（output_placeholders 等） | 负 token id 占位（FutureMap） | PLACEHOLDER_TOKEN 写入 outputTokenIds，再 Replace |
| **同步点显式化** | future.result() / structured output/spec 依赖点 | copy_done.synchronize / disable_overlap 条件 | launch_done、postprocess_done、ReplacePlaceHolder 的时序约束 |
| **可并发 outstanding batch** | batch_queue_size（PP 时>1） | 由 loop/队列深度与实现约束决定 | `maxScheduledBatch_`（由 activateAsyncInference/配置决定） |
| **复杂度/可维护性** | 中（抽象集中） | 高（性能强但约束多） | 高（C++/Python 双层异步 + 插件链） |

---

## 5) PlantUML

本目录附带 3 张图：
- `mindie_async_plugin_manager_pipeline.puml`
- `mindie_scheduler_placeholder_async.puml`
- `three_way_async_scheduling_compare.puml`

---

## 6) 三者优缺点（聚焦异步调度 / overlap / 跨步依赖）

### 6.1 vLLM（V1）

**优点**
- **抽象收敛、路径统一**：核心用 `Future +（PP 时）batch_queue` 推迟阻塞点；调度/执行边界清晰，通用 serving 场景适配面广。
- **并发度控制简单**：`batch_queue_size` 作为显式 knob，尤其对 PP“填流水/消 bubble”很直接。
- **跨步依赖更“状态化”**：更倾向用 request 的状态量（如 placeholder 计数、async scheduler 修正逻辑）表达“未落地 token”，少直接改写 token 序列。

**缺点**
- **系统级 overlap 不如 SGLang 激进**：整体更偏“批次/任务级”流水，CPU 后处理与 GPU 前向的细粒度重叠不一定最大化。
- **遇到 structured output / spec 等依赖仍会同步**：为了正确性会出现显式同步点，极限 overlap 会被依赖约束。

### 6.2 SGLang（SRT）

**优点**
- **系统工程化 overlap 很强**：`event_loop_overlap + forward_stream/event + non_blocking D2H + result_queue` 把“launch 当前 batch”与“处理上一轮结果”结构化交错。
- **依赖解析更高效**：`FutureMap` 用“负 token id 占位 + GPU resolve”解决跨 batch token 依赖，减少 CPU 等待。
- **缓存/长上下文场景优化丰富**：chunked prefill、cache-aware 排序等更容易在特定 workload 上打出优势。

**缺点**
- **模式分支多、正确性约束复杂**：例如 overlap 与 spec/grammar 的兼容性需要降级，维护成本较高。
- **强优化更依赖实现细节**：调参空间大，但边缘组合更容易引入性能回退或隐性同步点。

### 6.3 MindIE-LLM

**优点**
- **双层异步推进能力**：
  - C++ Scheduler：`PrepareNextSchedule()` 里 `AccumulateComputedTokens + AddNextTokenPlaceHolder(-1)`，支持 outstanding 多 batch；
  - Python PluginManager：`forward_thread + input/output queue + Event` 实现 “prev 后处理 + next forward 提交” 的 pipeline。
- **对 speculation/MTP 的资源约束意识强**：placeholder 数量有上限（与 `maxScheduledBatch_`、`1+gamma` 相关），避免长期占坑导致 KV 浪费。
- **插件扩展点丰富且直接**：preprocess/sample/verify/cache_update/cache_clear 等 hook 便于快速扩展能力。

**缺点**
- **跨语言双套流水并存，整体复杂度更高**：C++ placeholder 推进 + Python 线程 pipeline 同时存在，边界与一致性更难维护。
- **同步点偏“硬”**：Python 侧 `synchronize()` + `launch_done/postprocess_done` 双向等待，容易吞掉 overlap 的收益并增加尾延迟。
- **占位符直接写入 outputTokenIds**：直观但侵入性强，错误更容易变成“token 序列异常/占位符计数错误”，定位成本高。
- **异步能力受后端限制明显**：例如 async inference 直接要求 ATB backend，否则报错。

---

## 7) MindIE-LLM 相对 vLLM/SGLang 的可改进空间（按收益优先级）

### 7.1 收敛异步模型：避免“双套流水”叠加的复杂性
**现状**：C++ scheduler 的 placeholder 异步推进 + Python plugin_manager 的线程流水同时存在。  
**建议**：明确“一套主流水”的主导层（更偏 C++ 或更偏 Python），让另一层退化为轻量适配/插件编排，减少双向等待与状态重复。

### 7.2 降低 placeholder 对 token 序列的侵入（向 SGLang FutureMap 学习）
**现状**：PLACEHOLDER_TOKEN 写进 `outputTokenIds`，回填靠 `predictedTokensBySeqId_` 与 `ReplacePlaceHolderWithToken()`。  
**建议**：
- 把“未决 token”承载从 `outputTokenIds` 转为独立 buffer/ring（token 只在最终落地时更新），降低错误面与调试成本。
- 如果输入侧也存在“用上一轮 token 填下一轮输入”的需求，尽量避免全局同步后再填，优先考虑 device-side resolve 或更细粒度 event。

### 7.3 减少全局 `synchronize()` 与双向 Event 等待，让 overlap 真正生效
**现状**：`generate_token_async()` 的 `synchronize()`、以及 `launch_done/postprocess_done` 的双向等待会形成长链同步。  
**建议**：用更细粒度的 event/stream 或只同步必需数据，尽量把同步点集中到“消费结果”的边界，而不是贯穿流水线。

### 7.4 增加 backpressure 与队列深度治理（降低尾延迟与内存膨胀）
**现状**：Python `queue.Queue()` 默认无界；C++ 有 `maxScheduledBatch_` 但 Python pipeline 仍可能积压。  
**建议**：为 input/output queue 设置容量上限与降级策略（高压时退回同步、降低 outstanding），并结合 KV 空闲与 QPS 动态调节深度。

### 7.5 明确插件的异步安全契约（线程归属、读写顺序、可重入性）
**现状**：hook 多且强，async 模式下执行线程与时序更复杂。  
**建议**：框架层明确哪些 hook 必须在主线程/forward 线程/C++ 调度线程执行；哪些状态必须在“postprocess 前/后”写入，避免插件引入隐性数据竞争。

---

## 8) PlantUML（新增：MindIE 现状 vs 建议收敛方案）

- `mindie_pipeline_refactor_compare.puml`：MindIE 当前“双层异步”与“单一主流水”建议架构对比

