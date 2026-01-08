import time

import zmq


ENDPOINT = "tcp://127.0.0.1:5555"
TIMEOUT_MS = 500  # 0.5s 超时，方便你用“慢 server”触发重试
MAX_RETRIES = 5
BACKOFF_S = 0.2


def make_req_socket(ctx: zmq.Context) -> zmq.Socket:
    sock = ctx.socket(zmq.REQ)
    # 关闭时不要卡住（未发送完的消息直接丢弃）
    sock.setsockopt(zmq.LINGER, 0)
    # 发送/接收超时（否则默认会无限阻塞）
    sock.setsockopt(zmq.SNDTIMEO, TIMEOUT_MS)
    sock.setsockopt(zmq.RCVTIMEO, TIMEOUT_MS)
    sock.connect(ENDPOINT)
    return sock


def main():
    ctx = zmq.Context()
    sock = make_req_socket(ctx)

    try:
        for attempt in range(1, MAX_RETRIES + 1):
            msg = f"ping attempt={attempt} ts={time.time():.3f}"
            try:
                sock.send_string(msg)
                reply = sock.recv_string()
                print("[client] recv:", reply)
                return
            except zmq.Again:
                # REQ 的一个大坑：send 后没收到 reply 就不能再 send（状态机锁住）
                # 解决思路之一：超时后“重建 socket”，让状态机回到初始状态，再重试。
                print(f"[client] timeout on attempt {attempt}, retrying...")
                sock.close(linger=0)
                time.sleep(BACKOFF_S)
                sock = make_req_socket(ctx)

        raise TimeoutError(f"no reply after {MAX_RETRIES} attempts")
    finally:
        sock.close(linger=0)
        ctx.term()


if __name__ == "__main__":
    main()

