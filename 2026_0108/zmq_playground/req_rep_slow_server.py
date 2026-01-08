import argparse
import time

import zmq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", default="tcp://127.0.0.1:5555")
    parser.add_argument("--delay-s", type=float, default=1.5)
    parser.add_argument("--delay-first-n", type=int, default=2)
    args = parser.parse_args()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(args.bind)
    print(f"[server] REP bound at {args.bind}, delay={args.delay_s}s first_n={args.delay_first_n}")

    i = 0
    try:
        while True:
            msg = sock.recv_string()
            i += 1
            print(f"[server] recv({i}): {msg!r}")
            if i <= args.delay_first_n and args.delay_s > 0:
                print(f"[server] sleeping {args.delay_s}s to simulate slowness...")
                time.sleep(args.delay_s)
            sock.send_string(f"pong: {msg}")
    finally:
        sock.close(linger=0)
        ctx.term()


if __name__ == "__main__":
    main()

