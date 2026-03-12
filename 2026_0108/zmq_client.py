import zmq

def run_client():
    context = zmq.Context()

    # 使用 REQ (Request) 类型的 socket
    print("Connecting to hello world server...")
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")

    # 发送 10 次请求
    for request in range(10):
        print(f"Sending request {request} ...")
        # 发送请求
        socket.send(b"Hello")

        # 等待接收响应
        # recv 会阻塞直到收到响应
        message = socket.recv()
        print(f"Received reply {request} [ {message} ]")

if __name__ == "__main__":
    run_client()

