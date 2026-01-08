import threading
import time

import msgspec
import zmq


def frontend(router_addr: str, pull_addr: str) -> None:
    ctx = zmq.Context(io_threads=1)

    router = ctx.socket(zmq.ROUTER)
    router.bind(router_addr)

    pull = ctx.socket(zmq.PULL)
    pull.bind(pull_addr)

    # 等待 engine 先发一个空消息（vLLM 里也这么做：让 ROUTER 认识 identity）
    engine_id, empty = router.recv_multipart()
    assert empty == b""
    print(f"[frontend] engine connected: id={engine_id!r}")

    # 模拟 vLLM 的 (RequestType, SerializedRequest, aux_buffers...)
    request_type = b"ADD"  # vLLM 里是 EngineCoreRequestType.ADD.value (bytes)
    payload = {"request_id": "r1", "prompt": "hello", "ts": time.time()}
    payload_bytes = msgspec.msgpack.encode(payload)

    # 模拟一个“很大”的附加 buffer（vLLM 会用 msgspec + aux_buffers 做 zero-copy）
    aux = b"x" * (256 * 1024)

    router.send_multipart([engine_id, request_type, payload_bytes, aux], copy=False)
    print("[frontend] sent request")

    # 从引擎输出通道收结果（PUSH->PULL）
    out = pull.recv_multipart(copy=False)
    out_type = bytes(out[0].buffer)
    out_payload = msgspec.msgpack.decode(out[1].buffer)
    print(f"[frontend] got output: type={out_type!r}, payload={out_payload}")

    router.close(linger=0)
    pull.close(linger=0)
    ctx.term()


def engine(router_addr: str, pull_addr: str) -> None:
    ctx = zmq.Context(io_threads=1)

    dealer = ctx.socket(zmq.DEALER)
    dealer.setsockopt(zmq.IDENTITY, b"\x01\x00")  # 类似 vLLM：rank.to_bytes(2,"little")
    dealer.connect(router_addr)

    push = ctx.socket(zmq.PUSH)
    push.connect(pull_addr)

    # 先发空消息注册到 ROUTER
    dealer.send(b"")

    # 收请求：这里没有 ROUTER 的 identity 帧（DEALER 会自动剥离）
    req_type, payload_bytes, aux = dealer.recv_multipart(copy=False)
    req_type = bytes(req_type.buffer)
    payload = msgspec.msgpack.decode(payload_bytes.buffer)
    aux_len = len(aux.buffer)
    print(f"[engine] got request: type={req_type!r}, payload={payload}, aux={aux_len}")

    # 回一个简单输出（通过 PUSH->PULL）
    out_payload = {"request_id": payload["request_id"], "ok": True, "aux_len": aux_len}
    push.send_multipart([b"OUTPUT", msgspec.msgpack.encode(out_payload)], copy=False)

    dealer.close(linger=0)
    push.close(linger=0)
    ctx.term()


if __name__ == "__main__":
    # 用 tcp 最直观；你也可以改成 ipc:// 或 inproc://
    router_addr = "tcp://127.0.0.1:5560"
    pull_addr = "tcp://127.0.0.1:5561"

    t = threading.Thread(target=engine, args=(router_addr, pull_addr), daemon=True)
    t.start()
    time.sleep(0.1)

    frontend(router_addr, pull_addr)
    t.join(timeout=1.0)

