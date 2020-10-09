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
        they can sense all cells that are up to 5 steps away. """
        self.agent_vision = assumptions.VISION_RANGE

        self.action_space = spaces.Discrete(4) # NOT INCLUSIVE ## TODO

        # Current perception
        self.vision_size = np.zeros((vision_grid_size(self.agent_vision), 5) # Things, terrain
        self.agent_attached = np.zeros((vision_grid_size(assumptions.TASK_SIZE), 3) # Attached -> Extract attached type from lastAction + lastActionParameter
        self.forwarded_task = [''] * TASK_NUM # Names of the tracked tasks
        

        self.state = None # Observation_space instance
        

        #self.steps_beyond_done = None