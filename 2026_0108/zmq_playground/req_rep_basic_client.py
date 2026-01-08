import zmq


def main():
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect("tcp://127.0.0.1:5555")

    sock.send_string("ping-----------")
    print("[client] recv:", sock.recv_string())

    sock.close(linger=0)
    ctx.term()


if __name__ == "__main__":
    main()

