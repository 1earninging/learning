import threading
import time
from collections import deque

import msgspec
import zmq


TOPIC = b"kv"
END_SEQ = (-1).to_bytes(8, "big", signed=True)


class Server:
    def __init__(self, pub_addr: str, replay_addr: str):
        self.ctx = zmq.Context.instance()
        self.pub = self.ctx.socket(zmq.PUB)
        self.pub.bind(pub_addr)

        self.replay = self.ctx.socket(zmq.ROUTER)
        self.replay.bind(replay_addr)

        self.buf = deque[tuple[int, bytes]](maxlen=1000)
        self.seq = 0

    def publish_loop(self):
        while True:
            payload = {"seq": self.seq, "ts": time.time()}
            payload_bytes = msgspec.msgpack.encode(payload)
            seq_bytes = self.seq.to_bytes(8, "big")
            self.pub.send_multipart((TOPIC, seq_bytes, payload_bytes))
            self.buf.append((self.seq, payload_bytes))
            self.seq += 1
            time.sleep(0.05)

    def replay_loop(self):
        while True:
            # REQ -> ROUTER: [identity, b"", start_seq_bytes]
            ident, empty, start_seq_bytes = self.replay.recv_multipart()
            assert empty == b""
            start_seq = int.from_bytes(start_seq_bytes, "big")
            for seq, payload_bytes in self.buf:
                if seq >= start_seq:
                    self.replay.send_multipart(
                        (ident, b"", seq.to_bytes(8, "big"), payload_bytes)
                    )
            self.replay.send_multipart((ident, b"", END_SEQ, b""))


def subscriber(pub_addr: str, replay_addr: str):
    ctx = zmq.Context.instance()

    sub = ctx.socket(zmq.SUB)
    sub.connect(pub_addr)
    sub.setsockopt(zmq.SUBSCRIBE, TOPIC)

    req = ctx.socket(zmq.REQ)
    req.connect(replay_addr)

    # 先等几条实时消息
    for _ in range(5):
        topic, seq_bytes, payload_bytes = sub.recv_multipart()
        payload = msgspec.msgpack.decode(payload_bytes)
        print(f"[sub] live: {payload} (topic={topic!r}, seq={int.from_bytes(seq_bytes,'big')})")

    # 假装“掉线”错过了一些 seq，走 replay 从当前 seq-3 开始补
    need_from = max(0, int.from_bytes(seq_bytes, "big") - 3)
    req.send(need_from.to_bytes(8, "big"))
    while True:
        seq_b, payload_b = req.recv_multipart()
        if seq_b == END_SEQ:
            break
        payload = msgspec.msgpack.decode(payload_b)
        print(f"[sub] replay: {payload}")

    sub.close(linger=0)
    req.close(linger=0)


if __name__ == "__main__":
    pub_addr = "tcp://127.0.0.1:5570"
    replay_addr = "tcp://127.0.0.1:5571"

    server = Server(pub_addr, replay_addr)
    threading.Thread(target=server.publish_loop, daemon=True).start()
    threading.Thread(target=server.replay_loop, daemon=True).start()

    time.sleep(0.2)
    subscriber(pub_addr, replay_addr)

