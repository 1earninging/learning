import time

import zmq


ENDPOINT = "tcp://127.0.0.1:5555"
POLL_TIMEOUT_MS = 500
MAX_RETRIES = 5
BACKOFF_S = 0.2


def make_req_socket(ctx: zmq.Context) -> zmq.Socket:
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(ENDPOINT)
    return sock


def main():
    ctx = zmq.Context()
    sock = make_req_socket(ctx)
    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)

    try:
        for attempt in range(1, MAX_RETRIES + 1):
            msg = f"ping attempt={attempt} ts={time.time():.3f}"
            sock.send_string(msg)

            events = dict(poller.poll(POLL_TIMEOUT_MS))
            if events.get(sock) == zmq.POLLIN:
                reply = sock.recv_string()
                print("[client] recv:", reply)
                return

            # 超时：REQ 状态机已处于“等待回复”状态，不能直接再 send。
            # 这里用“重建 socket”把状态机重置，属于最易理解且稳定的方案。
            print(f"[client] poll timeout on attempt {attempt}, retrying...")
            poller.unregister(sock)
            sock.close(linger=0)
            time.sleep(BACKOFF_S)
            sock = make_req_socket(ctx)
            poller.register(sock, zmq.POLLIN)

        raise TimeoutError(f"no reply after {MAX_RETRIES} attempts")
    finally:
        poller.unregister(sock)
        sock.close(linger=0)
        ctx.term()


if __name__ == "__main__":
    main()

