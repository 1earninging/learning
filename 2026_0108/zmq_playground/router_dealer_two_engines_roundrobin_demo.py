import threading
import time

import msgspec
import zmq


ROUTER_ADDR = "tcp://127.0.0.1:5564"
PULL_ADDR = "tcp://127.0.0.1:5565"

ENGINE_COUNT = 2
TOTAL_REQUESTS = 10
POLL_TIMEOUT_MS = 1000


def engine(engine_index: int, router_addr: str, pull_addr: str) -> None:
    ctx = zmq.Context(io_threads=1)

    dealer = ctx.socket(zmq.DEALER)
    dealer.setsockopt(zmq.LINGER, 0)
    identity = engine_index.to_bytes(2, "little")
    dealer.setsockopt(zmq.IDENTITY, identity)
    dealer.connect(router_addr)

    push = ctx.socket(zmq.PUSH)
    push.setsockopt(zmq.LINGER, 0)
    push.connect(pull_addr)

    poller = zmq.Poller()
    poller.register(dealer, zmq.POLLIN)

    try:
        # 注册 identity 到 ROUTER（前端才能拿到 engine_id 列表）
        dealer.send(b"")
        print(f"[engine{engine_index}] registered identity={identity!r}")

        while True:
            events = dict(poller.poll(10_000))
            if events.get(dealer) != zmq.POLLIN:
                print(f"[engine{engine_index}] timeout waiting request, exit")
                return

            frames = dealer.recv_multipart(copy=False)
            # DEALER 侧看不到 identity，frames 是业务帧
            # 支持两种：
            # 1) STOP：一帧 [b"STOP"]
            # 2) 请求：>=2 帧 [type, payload, ...]
            msg_type = bytes(frames[0].buffer)
            if msg_type == b"STOP":
                print(f"[engine{engine_index}] got STOP, exit")
                return

            payload = msgspec.msgpack.decode(frames[1].buffer)
            req_id = payload["request_id"]
            print(f"[engine{engine_index}] handle req={req_id}")

            out_payload = {
                "request_id": req_id,
                "engine_index": engine_index,
                "engine_identity": identity.hex(),
                "ts": time.time(),
            }
            push.send_multipart([b"OUTPUT", msgspec.msgpack.encode(out_payload)], copy=False)
    finally:
        poller.unregister(dealer)
        dealer.close(linger=0)
        push.close(linger=0)
        ctx.term()


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

    engine_ids: list[bytes] = []
    sent = 0
    received = 0

    try:
        # 1) 等待所有 engine 注册 identity
        while len(engine_ids) < ENGINE_COUNT:
            events = dict(poller.poll(POLL_TIMEOUT_MS))
            if events.get(router) != zmq.POLLIN:
                print("[frontend] waiting engines...")
                continue
            engine_id, empty = router.recv_multipart()
            assert empty == b""
            if engine_id not in engine_ids:
                engine_ids.append(engine_id)
                print(f"[frontend] engine connected: {engine_id!r}")

        # 2) round-robin 发送请求（用 identity 选择目标 engine）
        for i in range(TOTAL_REQUESTS):
            target = engine_ids[i % len(engine_ids)]
            req_id = f"r{i}"
            payload = {"request_id": req_id, "i": i}
            router.send_multipart([target, b"ADD", msgspec.msgpack.encode(payload)], copy=False)
            print(f"[frontend] sent {req_id} -> {target!r}")
            sent += 1

        # 3) 汇聚输出（PUSH/PULL 不带 identity，所以 payload 里必须带 engine 标识）
        start = time.time()
        while received < sent:
            events = dict(poller.poll(POLL_TIMEOUT_MS))
            if events.get(pull) != zmq.POLLIN:
                if time.time() - start > 10:
                    raise TimeoutError("frontend timeout waiting outputs")
                continue
            out = pull.recv_multipart(copy=False)
            out_payload = msgspec.msgpack.decode(out[1].buffer)
            print(
                f"[frontend] got output req={out_payload['request_id']} "
                f"engine={out_payload['engine_index']} id={out_payload['engine_identity']}"
            )
            received += 1

        # 4) 停止所有 engine
        for eid in engine_ids:
            router.send_multipart([eid, b"STOP"], copy=False)
    finally:
        poller.unregister(router)
        poller.unregister(pull)
        router.close(linger=0)
        pull.close(linger=0)
        ctx.term()


if __name__ == "__main__":
    threads = []
    for idx in range(ENGINE_COUNT):
        t = threading.Thread(target=engine, args=(idx, ROUTER_ADDR, PULL_ADDR), daemon=True)
        t.start()
        threads.append(t)

    # 给 engine 一点时间 connect 并发送注册消息
    time.sleep(0.1)
    frontend(ROUTER_ADDR, PULL_ADDR)

    for t in threads:
        t.join(timeout=1.0)

