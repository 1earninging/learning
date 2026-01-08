# vLLM（OpenAI API server）一周冲刺：跑批脚本 + 统计报告

你选的是 **A：运行 vLLM OpenAI API server，然后用脚本打接口**。这一套目录给你：
- `batch_client.py`: 异步跑批（并发/限流/超时/重试），输出 JSONL
- `report.py`: 读取 JSONL，统计成功率、错误TopN、延迟分位数（P50/P95/P99）
- `prompts.txt`: 示例输入（每行一个 prompt）
- `requirements.txt`: 依赖（你用 pip 安装即可）

---

## 1) 安装依赖

在本目录执行：

```bash
python -m pip install -r requirements.txt
```

---

## 2) 先启动 vLLM OpenAI API server（示例）

你只要保证本机有一个 OpenAI 兼容的 endpoint，例如（具体命令以你自己的 vLLM 启动方式为准）：

- Base URL 通常是：`http://127.0.0.1:8000`
- Chat completions：`POST /v1/chat/completions`
- Completions：`POST /v1/completions`

> 说明：vLLM 的 OpenAI server 通常接受任意 API key（但你仍需要传一个）。

---

## 3) 跑批（非流式，推荐先用这个）

```bash
python batch_client.py \
  --base-url http://127.0.0.1:8000 \
  --api-key dummy \
  --mode chat \
  --model YOUR_MODEL_NAME \
  --input prompts.txt \
  --output out.jsonl \
  --concurrency 20 \
  --timeout 60 \
  --retries 2
```

跑完会生成 `out.jsonl`，每条记录包含：
- `ok`: 是否成功
- `latency_s`: 单条请求耗时（秒）
- `status_code` / `error_type` / `error_msg`
- `text`: 非流式时的最终文本（chat 为 assistant content；completion 为 choices[0].text）

---

## 4) 跑批（流式，进阶）

```bash
python batch_client.py \
  --base-url http://127.0.0.1:8000 \
  --api-key dummy \
  --mode chat \
  --model YOUR_MODEL_NAME \
  --input prompts.txt \
  --output out_stream.jsonl \
  --concurrency 10 \
  --timeout 60 \
  --retries 2 \
  --stream
```

流式模式会把收到的增量 token 拼成最终 `text`，并额外记录：
- `stream_chunks`: 收到的 SSE data chunk 数量（粗略反映流式切片频率）

---

## 5) 统计报告

```bash
python report.py --input out.jsonl
```

输出包括：
- 成功率
- 错误 TopN
- 延迟 P50/P95/P99

---

## 6) 一周（每天2小时）冲刺：读 vLLM + 写脚本（建议照做）

### Day 1：脚本“骨架能力” + 读码入口
- 读码（20m）：`vllm/v1/engine/async_llm.py` 顶部导入与类定义，先只看注释/Docstring，搞清 **AsyncLLM 是给 API server 用的**。
- 编码（70m）：能运行 `batch_client.py --help`，理解参数含义（base-url/model/input/output/concurrency/timeout/retries）。
- 回映（20m）：在 `AsyncLLM.generate()` 的 docstring 里找“流式输出/后台 loop”描述。
- 验收：你能用 3 句话描述：你的脚本要做什么、vLLM 服务端大概如何返回结果。

### Day 2：asyncio 最小核心（并发 vs 串行）
- 练习（核心）：看懂并写出 `Semaphore` 限流 + `gather` 并发。
- 回映：去看 `AsyncLLM._run_output_handler()`，理解为什么它会 `await asyncio.sleep(0)` 让出事件循环。
- 验收：你能解释“为什么 for 循环里逐个 await 仍然慢”。

### Day 3：异步生成器（对应 vLLM 流式）
- 练习：写一个最小 `async generator`（你可以直接读 `AsyncLLM.generate()` 的结构理解）。
- 回映：理解 `generate()` 为何 `yield out`，以及取消时为什么要 abort。
- 验收：你能解释“客户端断开/取消”在服务端要做什么清理。

### Day 4：超时、重试、错误分类（自动化必备）
- 练习：在跑批里实现 timeout + retries + 失败落盘（本仓库脚本已实现，你只要读懂并能改参数）。
- 回映：对照 vLLM 里各种异常分支（例如 EngineDead/ValueError）。
- 验收：你能从 out.jsonl 里统计失败原因分布。

### Day 5：IPC 心智模型（知道 vLLM 为什么拆进程）
- 读码：`vllm/v1/engine/core_client.py` 顶部注释（client/asyncio/multiprocessing）+ 类层次结构。
- 目标：一句话讲清：**AsyncLLM 在前端进程，EngineCore 在后台进程，靠 ZMQ/序列化传请求与输出**。
- 验收：画 5 步数据流图（请求进入→发到 core→core 输出→client 拉取→generate 流式返回）。

### Day 6：把脚本用起来：对你自己的 vLLM 服务做跑批
- 做事：准备一份真实的 `prompts.txt`（20~200 行），跑一次非流式，再跑一次流式。
- 验收：你能给出成功率、P95 延迟，并定位一类最常见错误（超时/连接失败/服务端 5xx）。

### Day 7：输出一份“读码报告”（非常重要）
- 产出：一页笔记（你自己的话）：
  - 关键文件：`async_llm.py`、`core_client.py`、`core.py`
  - 关键点：异步生成器、后台 task、取消/abort、IPC
  - 你仍不懂的 3 个点（下周继续）
- 验收：你能把 vLLM 主链路讲给另一个小白听（讲得通就算胜利）。

