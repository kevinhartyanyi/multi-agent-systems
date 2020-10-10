import server_env
import random, time
import socket, json
from message_classes import *
from action_classes import *
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import assumptions

directions = ["n", "s", "w", "e"]

class Reinforce_Agent(object):
    def __init__(self, name, id, env):
        self.name = name
        self.id = id
        self.env = env
        self.request_id = 0
        self.state = None
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.map = np.array([[0,0,0,0]])
        #self.last_action_parameter = [] # Needed?
        
        # Network parameters
        # self.state (vision_grid, agent_attached, forwarded_task, energy, step)
        self.known_dispensers = -1 * np.ones((assumptions.DISPENSER_NUM, 3))
        self.known_walls = -1 * np.ones((assumptions.WALL_NUM, 2))
        
    def act(self):
        state = torch.from_numpy(self.state).float().unsqueeze(0)
        probs = self.forward(Variable(state))
        highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        return highest_prob_action, log_prob

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
         

# Constants
GAMMA = 0.9

class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=3e-4):
        super(PolicyNetwork, self).__init__()

        self.num_actions = num_actions
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.softmax(self.linear2(x), dim=1)
        return x 
    
    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(Variable(state))
        highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        return highest_prob_action, log_prob
        
def update_policy(policy_network, rewards, log_probs):
    discounted_rewards = []

    for t in range(len(rewards)):
        Gt = 0 
        pw = 0
        for r in rewards[t:]:
            Gt = Gt + GAMMA**pw * r
            pw = pw + 1
        discounted_rewards.append(Gt)
        
    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9) # normalize discounted rewards

    policy_gradient = []
    for log_prob, Gt in zip(log_probs, discounted_rewards):
        policy_gradient.append(-log_prob * Gt)
    
    policy_network.optimizer.zero_grad()
    policy_gradient = torch.stack(policy_gradient).sum()
    policy_gradient.backward()
    policy_network.optimizer.step()
        
""" Notes

## We assume that:
    - vision distance = 5
    - number of block types = 3
    - we store a fixed number of tasks = 5
    - number of dispensers = 3
    - max energy = 300
    - number of steps = 500
    - task size is fixed = 1
"""