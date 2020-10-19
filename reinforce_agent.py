import random, time
import socket, json
import numpy as np
from message_classes import *
from action_classes import *
from utils import *
import assumptions
from policy_network import *

#pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

directions = ["n", "s", "w", "e"]

class Reinforce_Agent(object):
    def __init__(self, name, id, env):
        self.name = name
        self.id = id
        self.env = env
        self.request_id = 0
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.map = np.array([[0, 0, 0, 0, 0]]) # x, y, thing type, thing detail, terrain
        #self.last_action_parameter = [] # Needed?
        
        # Network parameters
        self.state = None #(vision_grid, agent_attached, forwarded_task, energy)
        #self.max_energy = np.array([assumptions.MAX_ENERGY])
        self.step = np.array([0, assumptions.STEP_NUM]) # current step, assumptions.STEP_NUM (Will certainly be updated, no need to set to assumptions.IGNORE)
        self.dispensers = assumptions.IGNORE * np.ones((assumptions.DISPENSER_NUM, 3))
        self.walls = assumptions.IGNORE * np.ones((assumptions.WALL_NUM, 2))
        
        # Policy Network
        num_inputs = self.env.get_state_size() + self.step.size + self.dispensers.size + self.walls.size
        print("NUM_INPUTS:", num_inputs)
        num_actions = len(action_dict)
        hidden_size = int((2/3) * num_inputs + num_actions)
        self.policy_network = PolicyNetwork(num_inputs, num_actions, hidden_size)
        
    def act(self):
        state = np.hstack([x.flatten() for x in self.state])
        #state += abs(np.amin(state))
        
        highest_prob_action, log_prob = self.policy_network.get_action(state)
        
        if 1 <= highest_prob_action <= 4: # Only need to update the map if we move
            self._update_coords(highest_prob_action)
        
        return highest_prob_action, log_prob

    def reset(self):
        self.request_id = 0

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.map = np.array([[0, 0, 0, 0, 0]])  # x, y, thing type, thing detail, terrain
        # self.last_action_parameter = [] # Needed?

        # Network parameters
        self.state = None  # (vision_grid, agent_attached, forwarded_task, energy)
        # self.max_energy = np.array([assumptions.MAX_ENERGY])
        self.step = np.array([0,
                              assumptions.STEP_NUM])  # current step, assumptions.STEP_NUM (Will certainly be updated, no need to set to assumptions.IGNORE)
        self.dispensers = assumptions.IGNORE * np.ones((assumptions.DISPENSER_NUM, 3))
        self.walls = assumptions.IGNORE * np.ones((assumptions.WALL_NUM, 2))

    def get_state(self):
        state = np.hstack([x.flatten() for x in self.state] + [0.5 for i in range(14)]).reshape((25,24))
        #print("State shape: ", state.shape)
        return state
        
    def _update_coords(self, direction: int):
        if direction == 1:
            self.map[:, 1] += 1
            self.walls[:,1][self.walls[:,1] != assumptions.IGNORE] += 1
            self.dispensers[:,1][self.dispensers[:,1] != assumptions.IGNORE] += 1
        elif direction == 2:
            self.map[:, 1] -= 1
            self.walls[:,1][self.walls[:,1]  != assumptions.IGNORE] -= 1
            self.dispensers[:,1][self.dispensers[:,1]  != assumptions.IGNORE] -= 1
        elif direction == 3:
            self.map[:, 0] -= 1
            self.walls[:,0][self.walls[:,0] != assumptions.IGNORE] -= 1
            self.dispensers[:,0][self.dispensers[:,0] != assumptions.IGNORE] -= 1
        elif direction == 4:
            self.map[:, 0] += 1
            self.walls[:,0][self.walls[:,0] != assumptions.IGNORE] += 1
            self.dispensers[:,0][self.dispensers[:,0] != assumptions.IGNORE] += 1
            
    def update_net(self, rewards, log_probs, retain):
        update_policy(self.policy_network, rewards, log_probs, retain)
        
    def update_env(self, msg):
        self.state = self.env.update(msg['content']['percept'])
        observation_vector = self.state[0]
        #print("Observation vector: ", observation_vector)
        for obs in observation_vector:
            ind = find_coord_index(self.map, obs[:2])
            # print("Check: ", obs[:2])
            # print("Index", ind)
            if ind == -1:  # New Entry
                self.map = np.append(self.map, np.array([obs]), axis=0)
            else:  # Update
                self.map[ind] = obs

        # Update walls
        new_walls = self.map[(self.map[:,2] == 0) & (self.map[:,3] == 0) & (self.map[:,4] == 2)][:,:2]

        empty_wall = np.where(self.walls[:,0] == assumptions.IGNORE)[0]
        new_walls_count = 0
        while new_walls_count < len(new_walls) and 0 < len(empty_wall):
            if new_walls[new_walls_count].tolist() not in self.walls.tolist():
                self.walls[empty_wall[0]] = new_walls[new_walls_count]
                empty_wall = empty_wall[1:]
            new_walls_count += 1
            
        # Update dispensers
        new_dispensers = self.map[(self.map[:,2] == 3)][:,:4]
        empty_dispensers = np.where(self.dispensers[:,0] == assumptions.IGNORE)[0]
        new_dispensers_count = 0
        while new_dispensers_count < len(new_dispensers) and 0 < len(empty_dispensers):
            if new_dispensers[new_dispensers_count][:2].tolist() not in self.dispensers[:,:2].tolist():
                self.dispensers[empty_dispensers[0]][:2] = new_dispensers[new_dispensers_count][:2]
                self.dispensers[empty_dispensers[0]][2] = new_dispensers[new_dispensers_count][3]
                empty_dispensers = empty_dispensers[1:]
            new_dispensers_count += 1

        # Final state of state ;)
        self.state = np.array([data for data in self.state] + [self.step, self.walls, self.dispensers])
        #print("State shape:", self.state.shape)
        
        # Visualization
        #self._visualize_map()
        #print("Current wall\n", self.walls)
        #print("Current dispensers\n", self.dispensers)
        #print("Current attached\n", self.state[1])
        
    def _visualize_map(self):
        minX = np.amin(self.map[:,0])
        maxX = np.amax(self.map[:,0])

        minY = np.amin(self.map[:,1])
        maxY = np.amax(self.map[:,1])

        cols = abs(minX) + maxX + 1
        rows = abs(minY) + maxY + 1

        print(rows, cols)


        things_type_map = np.zeros((rows, cols)) - 1
        things_details_map = np.zeros((rows, cols)) - 1
        terrain_map = np.zeros((rows, cols)) - 1

        for value in self.map:
            x, y = value[:2]
            x = x + abs(minX)
            y = y + abs(minY)
            things_type_map[y, x] = value[2]
            things_details_map[y, x] = value[3]
            terrain_map[y, x] = value[4]

        print("Map shape: ", things_type_map.shape)

        print("Things type map: ")
        print(things_type_map)

        print("Things details map: ")
        print(things_details_map)

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
                time.sleep(0.1)
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

    def send(self, action: int):
        agent_message = ActionReply(self.request_id, action_dict[action])
        msg = agent_message.msg()
        # print(f"Sending: {msg}")
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
            self.step[0] = response['content']['step'] # Current steps
            #self.update_env(response)
        ##TODO Check request_action
        return response
         

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