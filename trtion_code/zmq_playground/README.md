# ZMQ 入门（对照 vLLM 的用法）

这里放了两个最小 demo，目标是让你能快速看懂 vLLM 里 ZMQ 的几种关键模式，并能自己改协议/改 socket 参数做验证。

## 环境准备

```bash
python3 -m pip install pyzmq msgspec
```

## Demo 1：vLLM 风格的 ROUTER/DEALER + PULL/PUSH

这个 demo 模拟 `vllm/v1/engine/core_client.py` 和 `vllm/v1/engine/core.py` 的基本通信拓扑：

- **前端**：`ROUTER`（接收/路由请求） + `PULL`（接收输出）
- **引擎**：`DEALER`（带 identity） + `PUSH`（发输出到前端的 PULL）

运行：

```bash
python3 router_dealer_pushpull_demo.py
```

## Demo 2：PUB/SUB + ROUTER Replay（类似 kv_events）

模拟 `vllm/distributed/kv_events.py` 里的思路：

- PUB 按 `(topic, seq, payload)` 发事件
- SUB 订阅 topic
- 通过一个 ROUTER 额外提供 replay：SUB 用 REQ 请求从某个 seq 开始补发

运行：

```bash
python3 pubsub_replay_demo.py
```

