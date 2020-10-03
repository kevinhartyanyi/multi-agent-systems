import server_env
import random, time
import socket, json
from message_classes import *
from action_classes import *
import numpy as np

directions = ["n", "s", "w", "e"]

class Random_Agent(object):
    def __init__(self, name, id, env):
        self.name = name
        self.id = id
        self.env = env
        self.request_id = 0
        self.mem = []
        self.state = None
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.map = np.array([])
		
    def act(self):
        ind = random.randint(0, len(directions) - 1)
        high_level_thinking = directions[ind]
        action = ActionMove(high_level_thinking)
        return action, ind
        
    def update_env(self, msg):
        self.state = self.env.update(msg['content']['percept'])
        #return self.act(state)
        
    def connect(self, host: str = "127.0.0.1", port: int = 12300):
        connected = False
        wait_sec = 2
        while not connected:
            try:
                self.sock.connect((host, port))
            except ConnectionRefusedError:
                print(f"Connection Refused. Trying again after {wait_sec} seconds")
                time.sleep(wait_sec)
            else:
                connected = True

    def init_agent(self):
        agent_message = AuthRequest(self.name, self.id)
        msg = agent_message.msg()
        print(f"Sending: {msg}")
        self.sock.sendall(msg.encode())
        response = json.loads(self.sock.recv(4096).decode("ascii").rstrip('\x00'))
        print(f"Response: {response}")
        return True if response["content"]["result"] == "ok" else False

    def send(self, action):
        agent_message = ActionReply(self.request_id, action)
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
        if response['type'] == "request-action":
            self.request_id = response['content']['id']
            #self.update_env(response)
        ##TODO Check request_action
        return response