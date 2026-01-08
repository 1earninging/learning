# ZMQ 入门（对照 vLLM 的用法）

这里放了两个最小 demo，目标是让你能快速看懂 vLLM 里 ZMQ 的几种关键模式，并能自己改协议/改 socket 参数做验证。

## 环境准备

```bash
python3 -m pip install pyzmq msgspec
```

## 推荐学习顺序（先基础再看 vLLM 风格）

1. `REQ/REP`：理解“请求-响应”和阻塞/时序约束
2. `PUB/SUB`：理解订阅时序、topic、多订阅者
3. `PUSH/PULL`：理解任务分发与回传
4. `ROUTER/DEALER`：理解 identity 路由与 multipart（vLLM 主要用它）

## Demo 1：最小 REQ/REP（入门必做）

开两个终端：

```bash
python3 req_rep_basic_server.py
```

```bash
python3 req_rep_basic_client.py
```

你会看到 client 打印 `pong: ping`，server 端会打印收到的消息。

## Demo 1.1：REQ/REP 超时 + 重试（工程必备）

**为什么需要它**：`recv()` 默认会一直等；一旦对端慢/断线，程序就会卡死。并且 **REQ socket 有严格状态机**：`send` 之后必须 `recv`，超时后如果你直接再 `send`，会出问题。

这个 demo 用的策略是：**超时后重建 REQ socket**，让状态机回到初始状态再重试（简单可靠，便于理解）。

开两个终端：

```bash
python3 req_rep_slow_server.py --delay-s 1.5 --delay-first-n 2
```

```bash
python3 req_rep_timeout_retry_client.py
```

你会看到 client 前几次超时重试，随后拿到一次成功回复。

## Demo 1.2：用 Poller 做超时（更贴近 vLLM 的写法）

**核心思路**：不用 `RCVTIMEO`，而是用 `zmq.Poller().poll(timeout_ms)` 监听 socket 是否可读（`POLLIN`）。这在你后面要处理 **多个 socket**（例如 vLLM 的 input/output/control）时非常关键。

运行方式同样是开两个终端（可继续用慢 server）：

```bash
python3 req_rep_slow_server.py --delay-s 1.5 --delay-first-n 2
```

```bash
python3 req_rep_poller_client.py
```

你会看到 client 输出 `poll timeout ... retrying...`，直到收到一次成功回复。

## Demo 2：vLLM 风格的 ROUTER/DEALER + PULL/PUSH

这个 demo 模拟 `vllm/v1/engine/core_client.py` 和 `vllm/v1/engine/core.py` 的基本通信拓扑：

- **前端**：`ROUTER`（接收/路由请求） + `PULL`（接收输出）
- **引擎**：`DEALER`（带 identity） + `PUSH`（发输出到前端的 PULL）

运行：

```bash
python3 router_dealer_pushpull_demo.py
```

## Demo 2.1：ROUTER/DEALER + Poller（更贴近 vLLM 多 socket 事件循环）

这个版本会用 `zmq.Poller` **同时监听**：

- `ROUTER`：等引擎注册 identity / 收控制类消息
- `PULL`：等引擎输出

运行：

```bash
python3 router_dealer_poller_demo.py
```

## Demo 2.2：2 个 engine + round-robin 路由（看清“发给谁”）

这个 demo 会启动 **2 个 engine**，它们分别设置不同的 `IDENTITY`（例如 `b"\\x00\\x00"`、`b"\\x01\\x00"`）。

前端会：

- 先收集两个 engine 的 identity
- 再把 `r0..r9` 这 10 个请求按 round-robin 轮询发给两个 engine
- 最后从 `PULL` 汇聚输出（因为 `PUSH/PULL` 不带 identity，所以输出 payload 里会带上 `engine_index/engine_identity`）

运行：

```bash
python3 router_dealer_two_engines_roundrobin_demo.py
```

## Demo 3：PUB/SUB + ROUTER Replay（类似 kv_events）

模拟 `vllm/distributed/kv_events.py` 里的思路：

- PUB 按 `(topic, seq, payload)` 发事件
- SUB 订阅 topic
- 通过一个 ROUTER 额外提供 replay：SUB 用 REQ 请求从某个 seq 开始补发

运行：

```bash
python3 pubsub_replay_demo.py
```

