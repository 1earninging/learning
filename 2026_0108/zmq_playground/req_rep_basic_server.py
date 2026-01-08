import zmq


def main():
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind("tcp://127.0.0.1:5555")
    print("[server] REP bound at tcp://127.0.0.1:5555")

    try:
        while True:
            msg = sock.recv_string()
            print(f"[server] recv: {msg!r}")
            sock.send_string(f"pong: {msg}")
    finally:
        sock.close(linger=0)
        ctx.term()


if __name__ == "__main__":
    main()

