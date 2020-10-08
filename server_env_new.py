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

## Constants
TASK_NUM = 5

class Server():
    def __init__(self):
        """Agents perceive the state of a cell depending on their vision. E.g. if they have a vision of 5, 
        they can sense all cells that are up to 5 steps away. """
        self.agent_vision = assumptions.VISION_RANGE ##TODO

        self.action_space = spaces.Discrete(4) # NOT INCLUSIVE

        self.vision_size = vision_grid_size(5)

        low = np.zeros((self.vision_size, 4))
        high = np.zeros((self.vision_size, 4))
        low[:, :2] = -self.agent_vision
        high[:, :2] = self.agent_vision
        high[:, 2:3] = 2  # cell type -> high is inclusive
        high[:, 3:] = 4  # thing type

        # Other alternative: spaces.Dict()
        """
        shape = (x-1, 2)
            x is the amount of cells that agent can see
            the second dimension contains information about the cell:
                - [0] the type of the cell (empty if not specified)
                - [1] the thing on the cell (if any)
            low,high: Independent bounds for each dimension
        """
        """ Observation_space
            - lastActionParams: Discrete(4)
            - lastAction: Discrete(1) ##TODO
            - lastActionResult: Discrete(2) # Failed or not
            - things/terrain: Box((vision_size,2)) ##TODO no details included
            * score
            * attached
            * energy
            * task
        """
        ## NOPE
        self.observation_space = spaces.Tuple((
            spaces.Discrete(4),
            spaces.Discrete(1),
            spaces.Discrete(2),
            spaces.Box(
            low=low,
            high=high,
            shape=((self.vision_size, 4)),
            dtype=np.int
            )
        ))

        # self.seed()
        # self.viewer = None
        self.state = None # Observation_space instance
        

        #self.steps_beyond_done = None