import json
import socket




class Server:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connect()

    def connect(self, host: str = "127.0.0.1", port: int = 12300):
        self.sock.connect((host, port))

    def init_agent(self, agent_message):
        msg = agent_message.msg()
        print(f"Sending: {msg}")
        self.sock.sendall(msg.encode())
        response = json.loads(self.sock.recv(4096).decode("ascii").rstrip('\x00'))
        print(f"Response: {response}")
        return True if response["content"]["result"] == "ok" else False

    def send(self, agent_message):
        msg = agent_message.msg()
        print(f"Sending: {msg}")
        self.sock.sendall(msg.encode())

    def receive(self):
        while True:
            recv = self.sock.recv(4096).decode("ascii").rstrip('\x00')
            if recv != "":
                break
        response = json.loads(recv.rstrip('\x00'))
        print(f"Response: {response}")
        return response

