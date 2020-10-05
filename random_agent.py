import server_env
import random, time
import socket, json
from message_classes import *
from action_classes import *
import numpy as np
from server_env import find_ind_in_observation_np_array

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
        self.map = np.array([[0,0,0,0]])
		
    def act(self):
        ind = random.randint(0, len(directions) - 1)
        high_level_thinking = directions[ind]
        action = ActionMove(high_level_thinking)

        self.update_map(high_level_thinking)

        return action, ind

    def update_map(self, direction):
        print("Direction: ", direction)
        # TODO: Maybe switch the values???
        if direction == "n":
            self.map[:, 1] += 1
        elif direction == "w":
            self.map[:, 0] += 1
        elif direction == "s":
            self.map[:, 1] -= 1
        elif direction == "e":
            self.map[:, 0] -= 1

    def update_env(self, msg):
        self.state = self.env.update(msg['content']['percept'])
        observation_vector = self.state[3]
        #print("Observation vector: ", observation_vector)
        for obs in observation_vector:
            ind = find_ind_in_observation_np_array(self.map, obs[:2])
            #print("Check: ", obs[:2])
            #print("Index", ind)
            if ind == -1: # New Entry
                self.map = np.append(self.map, np.array([obs]), axis=0)
            else: # Update
                self.map[ind] = obs
        self.visualize_map()
        #return self.act(state)

    def visualize_map(self):
        minX = np.amin(self.map[:,0])
        maxX = np.amax(self.map[:,0])

        minY = np.amin(self.map[:,1])
        maxY = np.amax(self.map[:,1])

        cols = abs(minX) + maxX + 1
        rows = abs(minY) + maxY + 1

        print(rows, cols)


        things_map = np.zeros((rows, cols)) - 1
        terrain_map = np.zeros((rows, cols)) - 1

        for value in self.map:
            x,y = value[:2]
            x = x + abs(minX)
            y = y + abs(minY)
            things_map[y,x] = value[2]
            terrain_map[y,x] = value[3]

        print("Map shape: ", things_map.shape)

        print("Things map: ")
        print(things_map)

        print("Terrain map: ")
        print(terrain_map)



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
        #print(f"Sending: {msg}")
        self.sock.sendall(msg.encode())
        response = json.loads(self.sock.recv(4096).decode("ascii").rstrip('\x00'))
        #print(f"Response: {response}")
        return True if response["content"]["result"] == "ok" else False

    def send(self, action):
        agent_message = ActionReply(self.request_id, action)
        msg = agent_message.msg()
        #print(f"Sending: {msg}")
        self.sock.sendall(msg.encode())

    def receive(self):
        while True:
            recv = self.sock.recv(4096).decode("ascii").rstrip('\x00')
            if recv != "":
                break
        response = json.loads(recv.rstrip('\x00'))
        #print(f"Response: {response}")
        if response['type'] == "request-action":
            self.request_id = response['content']['id']
            #self.update_env(response)
        ##TODO Check request_action
        return response