import threading
import time

import msgspec
import zmq


POLL_TIMEOUT_MS = 1000


def frontend(router_addr: str, pull_addr: str) -> None:
    ctx = zmq.Context(io_threads=1)

    router = ctx.socket(zmq.ROUTER)
    router.setsockopt(zmq.LINGER, 0)
    router.bind(router_addr)

    pull = ctx.socket(zmq.PULL)
    pull.setsockopt(zmq.LINGER, 0)
    pull.bind(pull_addr)

    poller = zmq.Poller()
    poller.register(router, zmq.POLLIN)
    poller.register(pull, zmq.POLLIN)

    engine_id: bytes | None = None
    got_output = False

    try:
        start = time.time()
        while True:
            events = dict(poller.poll(POLL_TIMEOUT_MS))
            if not events:
                if time.time() - start > 10:
                    raise TimeoutError("frontend timeout waiting for events")
                print("[frontend] poll timeout, waiting...")
                continue

            if engine_id is None and events.get(router) == zmq.POLLIN:
                # ROUTER 收到的第一帧是对端 identity
                # 我们让 engine 先发一个空消息，让 ROUTER 记录这个 identity
                engine_id, empty = router.recv_multipart()
                assert empty == b""
                print(f"[frontend] engine connected: id={engine_id!r}")

                # 发请求（multipart）：[identity, type, payload, aux]
                request_type = b"ADD"  # 对齐 vLLM：EngineCoreRequestType.ADD.value (bytes)
                payload = {"request_id": "r1", "prompt": "hello", "ts": time.time()}
                payload_bytes = msgspec.msgpack.encode(payload)
                aux = b"x" * (256 * 1024)

                router.send_multipart(
                    [engine_id, request_type, payload_bytes, aux], copy=False
                )
                print("[frontend] sent request")

            if events.get(pull) == zmq.POLLIN:
                out = pull.recv_multipart(copy=False)
                out_type = bytes(out[0].buffer)
                out_payload = msgspec.msgpack.decode(out[1].buffer)
                print(f"[frontend] got output: type={out_type!r}, payload={out_payload}")
                got_output = True

            if engine_id is not None and got_output:
                return
    finally:
        poller.unregister(router)
        poller.unregister(pull)
        router.close(linger=0)
        pull.close(linger=0)
        ctx.term()


def engine(router_addr: str, pull_addr: str) -> None:
    ctx = zmq.Context(io_threads=1)

    dealer = ctx.socket(zmq.DEALER)
    dealer.setsockopt(zmq.LINGER, 0)
    dealer.setsockopt(zmq.IDENTITY, b"\x01\x00")  # 类似 vLLM：rank.to_bytes(2,"little")
    dealer.connect(router_addr)

    push = ctx.socket(zmq.PUSH)
    push.setsockopt(zmq.LINGER, 0)
    push.connect(pull_addr)

    poller = zmq.Poller()
    poller.register(dealer, zmq.POLLIN)

    try:
        # 先发空消息注册到 ROUTER（让前端能拿到 identity）
        dealer.send(b"")

        # 等待前端发来请求
        events = dict(poller.poll(5000))
        if events.get(dealer) != zmq.POLLIN:
            raise TimeoutError("engine timeout waiting for request")

        # DEALER 收到的帧里没有 identity（被 ROUTER/DEALER 模式隐藏了）
        req_type, payload_bytes, aux = dealer.recv_multipart(copy=False)
        req_type = bytes(req_type.buffer)
        payload = msgspec.msgpack.decode(payload_bytes.buffer)
        aux_len = len(aux.buffer)
        print(f"[engine] got request: type={req_type!r}, payload={payload}, aux={aux_len}")

        out_payload = {"request_id": payload["request_id"], "ok": True, "aux_len": aux_len}
        push.send_multipart([b"OUTPUT", msgspec.msgpack.encode(out_payload)], copy=False)
    finally:
        poller.unregister(dealer)
        dealer.close(linger=0)
        push.close(linger=0)
        ctx.term()


if __name__ == "__main__":
    router_addr = "tcp://127.0.0.1:5562"
    pull_addr = "tcp://127.0.0.1:5563"

    t = threading.Thread(target=engine, args=(router_addr, pull_addr), daemon=True)
    t.start()
    time.sleep(0.1)

    frontend(router_addr, pull_addr)
    t.join(timeout=1.0)

