import time
import zmq

def run_server():
    context = zmq.Context()
    # 使用 REP (Reply) 类型的 socket
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")

    print("Server started on port 5555...")

    while True:
        # 等待客户端请求
        # recv 会阻塞直到收到消息
        message = socket.recv()
        print(f"Received request: {message}")

        # 模拟一些工作（例如处理请求）
        time.sleep(1)

        # 发送响应
        # send 发送消息回客户端
        socket.send(b"World")

if __name__ == "__main__":
    try:
        run_server()
    except KeyboardInterrupt:
        print("Server stopped.")

