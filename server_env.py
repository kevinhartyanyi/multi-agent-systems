import json
import socket
from message_classes import *
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import time

"""Example: 
        {'lastActionParams': [], 'score': 0, 'lastAction': '', 'things': [{'x': 0, 'y': 0, 'details': 
        'A', 'type': 'entity'}], 'attached': [], 'disabled': False, 'terrain': {'goal': [[0, 4], [-1, 4], [0, 5]]}, 
        'lastActionResult': '', 'tasks': [], 'energy': 300} """

"""{'lastActionParams': ['w'], 'score': 0, 'lastAction': 'move', 'things': [{'x': 0, 'y': 0, 'details': 'A', 
'type': 'entity'}], 'attached': [], 'disabled': False, 'terrain': {'obstacle': [[3, 2], [2, 2], [1, 2], [0, 2], [-1, 
2], [-2, 2], [-5, 0], [-3, 2], [1, -4], [0, -4], [-1, -4]]}, 'lastActionResult': 'success', 'tasks': [{'reward': 10, 
'requirements': [{'x': 0, 'y': 1, 'details': '', 'type': 'b2'}], 'name': 'task11', 'deadline': 408}, """

"""
"clearEnergyCost" : 50
"disableDuration" : 4,
"maxEnergy" : 300,
"attachLimit" : 10

"grid" : {
        "height" : 40,
        "width" : 40,
        "file" : "conf/maps/test40x40.bmp"
      },


"blockTypes" : [3, 3],
"dispensers" : [2, 3]

"tasks" : {
        "size" : [1, 1],
        "duration" : [100, 200],
        "probability" : 0.05
      }
"""


def vision_grid_size(vision):
    """
    Calculates the maximum visible cell amount based on vision.
    """
    if vision > 0:
        return 4 * vision + vision_grid_size(vision - 1)
    else:
        return 0


class Server(gym.Env):
    """
    Description:
        Handles the communication with the server and manages the environment for the agents.

    Actions:
        Type: Discrete(4) TODO: Implement the other actions
        Num   Action
        0     move north
        1     move east
        2     move south
        3     move west

    Observation:
        Type: Box(vision_size, 2)
        Num             Dim        Observation               Min                     Max
        0               0           Cell Type                 0                       3
        0               1           Thing Type                0                       4
        1               --          --                        --                      --
        ...             --          --                        --                      --
        vision_size     --          --                        --                      --

        Cell Types:
            Num   Type
            0     Empty
            1     Obstacle
            2     Goal

        Thing Types:
            Num   Type
            0     Nothing
            1     Entity
            2     Block
            3     Dispenser
            4     Marker

    Reward:
        Currently using the 'score' returned by the server in every request-action message.
        TODO: More robust reward implementation

    Starting State:
        All observations are from the server

    Episode Termination:
        Server returns 'sim-end' message.
        TODO: More stopping points (maybe)

    """

    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connect()

        """Agents perceive the state of a cell depending on their vision. E.g. if they have a vision of 5, 
        they can sense all cells that are up to 5 steps away. """
        self.agent_vision = 5

        self.action_space = spaces.Discrete(4)

        vision_size = vision_grid_size(5)

        low = np.zeros(vision_size, 2)
        high = np.zeros(vision_size, 2)
        high[:, 0] = 2  # cell type
        high[:, 1] = 4  # thing type

        # Other alternative: spaces.Dict()
        """
        shape = (x-1, 2)
            x is the amount of cells that agent can see
            the second dimension contains information about the cell:
                - [0] the type of the cell (empty if not specified)
                - [1] the thing on the cell (if any)
            low,high: Independent bounds for each dimension
        """
        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            shape=(vision_size, 2),
            dtype=np.int
        )

        # self.seed()
        # self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        pass

    def reset(self):
        pass

    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Args:
            action (object): an action provided by the agent

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        if action == 0:
            #cell_type, thing = self.state[]
            pass
        elif action == 1:
            pass
        elif action == 2:
            pass
        elif action == 3:
            pass

        return np.array(self.state), reward, done, {}

    def render(self, mode='human'):
        pass

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

    def init_agent(self, agent_message: AuthRequest):
        msg = agent_message.msg()
        print(f"Sending: {msg}")
        self.sock.sendall(msg.encode())
        response = json.loads(self.sock.recv(4096).decode("ascii").rstrip('\x00'))
        print(f"Response: {response}")
        return True if response["content"]["result"] == "ok" else False

    def send(self, agent_message: ActionReply):
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
