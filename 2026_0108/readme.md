# Python ZeroMQ (pyzmq) 入门

## 1. 安装

首先需要安装 `pyzmq` 库：

```bash
pip install pyzmq
```

## 2. 核心概念

ZeroMQ (ZMQ) 是一个高性能的异步消息库，用于分布式或并发应用。它提供了比 TCP socket 更高级的抽象。

### 关键组件

- **Context (`zmq.Context`)**: 创建 Socket 的容器，通常一个进程一个 Context。
- **Socket (`zmq.Socket`)**: 用于发送和接收消息。ZMQ 的 Socket 是特定类型的（Pattern）。
- **Pattern (模式)**: ZMQ 定义了多种消息模式，最常见的是：
    - **Request-Reply (REQ-REP)**: 客户端发送请求，服务器回应。同步的，一问一答。
    - **Pub-Sub (PUB-SUB)**: 发布者广播，订阅者接收。异步的。
    - **Push-Pull (PUSH-PULL)**: 管道模式，用于任务分发。

## 3. 示例：Request-Reply (Hello World)

这个示例展示了最基础的同步通信。

- **Server (`zmq_server.py`)**:
    - 使用 `zmq.REP` (Reply) socket。
    - 绑定到端口 `5555`。
    - 循环：接收 "Hello" -> 等待 1秒 -> 发送 "World"。

- **Client (`zmq_client.py`)**:
    - 使用 `zmq.REQ` (Request) socket。
    - 连接到端口 `5555`。
    - 发送 "Hello" -> 等待接收 "World"。

### 运行方法

你需要打开两个终端窗口。

**终端 1 (Server):**
```bash
python zmq_server.py
```

**终端 2 (Client):**
```bash
python zmq_client.py
```

