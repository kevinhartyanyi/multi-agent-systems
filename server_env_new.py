import json
import socket
from message_classes import *
#import gym
#from gym import spaces, logger
#from gym.utils import seeding
import numpy as np
import time
import subprocess
import itertools
import assumptions
from utils import *

## Constants
TASK_NUM = 5

class Server():
    def __init__(self):
        """Agents perceive the state of a cell depending on their vision. E.g. if they have a vision of 5, 
        they can sense all cells that are up to 5 steps away.
        """
        self.agent_vision = assumptions.VISION_RANGE

        #self.action_space = spaces.Discrete(4) # NOT INCLUSIVE ## TODO

        # Current perception
        self.vision_grid = -1 * np.ones((vision_grid_size(self.agent_vision), 5) # Things, terrain
        self.agent_attached = np.zeros((vision_grid_size(assumptions.TASK_SIZE), 2) # Attached -> Extract attached type from lastAction + lastActionParameter
        self.forwarded_task_names = [''] * TASK_NUM # Names of the tracked tasks
        self.forwarded_task = -1 * np.ones((TASK_NUM, (2 + assumptions.TASK_SIZE * 3)))
        
        self.energy = 0
        self.step = 0
        
        self.disabled = False
        self.lastActionResult = 'success'
        

        self.state = None # Observation_space instance (vision_grid, agent_attached, forwarded_task, energy, step)
        
    def reset(self):
        ## TODO Start server (subprocess)
        return None
        
    def get_task_name(self, index):
        return self.forwarded_task_names[index]
        
    def step(self, action: int):
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Args:
            action (int): an action provided by the agent

        Returns:
            state (tuple): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
        """
        action_class = action_dict[action]
        reward = 0
        
        return self.state, reward
        
    def update(self, msg):
        pass