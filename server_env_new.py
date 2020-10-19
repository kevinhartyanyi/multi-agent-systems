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


class Server():
    def __init__(self):
        """Agents perceive the state of a cell depending on their vision. E.g. if they have a vision of 5, 
        they can sense all cells that are up to 5 steps away.
        """
        self.agent_vision = assumptions.VISION_RANGE

        #self.action_space = spaces.Discrete(4) # NOT INCLUSIVE ## TODO

        # Current perception
        self.vision_grid = assumptions.IGNORE * np.ones((vision_grid_size(self.agent_vision)+1, 5)) # Things, terrain
        self.agent_attached = assumptions.IGNORE * np.ones((vision_grid_size(assumptions.TASK_SIZE), 2)) # Attached -> Extract attached type from lastAction + lastActionParameter
        self.forwarded_task_names = [str(assumptions.IGNORE)] * assumptions.TASK_NUM # Names of the tracked tasks
        self.forwarded_task = assumptions.IGNORE * np.ones((assumptions.TASK_NUM, (2 + assumptions.TASK_SIZE * 3))) # x, y, deadline, points, block_num
        
        self.energy = np.array([assumptions.IGNORE])
        #self.step = 0 # Not in percept
        
        self.disabled = False
        self.lastActionResult = 'success'
        

        self.state = None # Observation_space instance (vision_grid, agent_attached, forwarded_task, energy)
        
    def reset(self):
        self.__init__()
        
    def get_state_size(self):
        return self.vision_grid.size + self.agent_attached.size + self.forwarded_task.size + self.energy.size
        
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
        # Things and Terrain
        observation_map = init_vision_grid(self.agent_vision)
        things = msg['things']
        terrain = msg['terrain']
        # print(f"\n\n\nThings: {things}")
        # print(f"Terrain: {terrain}")
        self.energy[0] = msg['energy']
        attached = msg['attached'] # List of coordinates
        
        if len(attached) > 0:
           
            # Update agent_attached
            attached = np.asarray(attached)
            size_diff = self.agent_attached.shape[0] - attached.shape[0] 
            if size_diff > 0:
                self.agent_attached = np.vstack([attached, assumptions.IGNORE * np.ones((size_diff, 2)) ])
            else:
                self.agent_attached = attached[:self.agent_attached.shape[0], :] # Just a precaution, in case agent_attached isnt large enough
         
        for th in things:
            x = th["x"]
            y = th["y"]
            detail = th["details"]
            typ = th["type"]

            ind = find_coord_index(observation_map, [x, y])

            # print(f"Thing detail: {detail}")
            # print(f"Map ind: {ind}")

            observation_map[ind][2] = get_things_code(typ)
            observation_map[ind][3] = get_things_details(typ, detail)

        terrain_values = ["goal", "obstacle"]

        for name in terrain_values:
            try:
                terran_cords = terrain[name]
                # print(f"Terrain cords: {terran_cords}")
                for cords in terran_cords:
                    x, y = cords
                    ind = find_coord_index(observation_map, [x, y])
                    observation_map[ind][4] = get_terrain_code(name)
                    #print("Terrain cords: ", cords)
            except:
                #print(f"Terrain: {name} not found")
                pass


        self.vision_grid = np.asarray(observation_map)



        # Tasks
        tasks = msg["tasks"]
        preprocessed_tasks = []

        for t in tasks:
            # print("Requirements size: ", len(t["requirements"]))
            # if len(t["requirements"]) > 1:
            #    input()
            points = t["reward"]
            requirements = t["requirements"][0]  # TODO: Only using the first requirement
            name = t["name"]  # TODO: Currently not used
            deadline = t["deadline"]
            details = requirements["details"]  # TODO: Find a use for this
            x = requirements["x"]
            y = requirements["y"]
            block = requirements["type"]
            




            # Convert block
            block_num = get_things_details("block", block)

            preprocessed_tasks.append([
                name, x, y, deadline, points, block_num
            ])

        #print("Preprocessed Tasks: \n", preprocessed_tasks)

        # Check if stored task is still active
        task_names = [t[0] for t in preprocessed_tasks]
        for i, name in enumerate(self.forwarded_task_names):
            if name not in task_names and name != str(assumptions.IGNORE):  # Delete if task is over
                self.forwarded_task[i] = assumptions.IGNORE * np.ones(2 + assumptions.TASK_SIZE * 3)
            elif name in task_names:  # Update otherwise
                for t in preprocessed_tasks:
                    if t[0] == name:
                        self.forwarded_task[i] = np.asarray(t[1:])
                        break


        free_places = [i for i, n in enumerate(self.forwarded_task_names) if n == str(assumptions.IGNORE)]
        not_stored_yet = [i for i, n in enumerate(preprocessed_tasks) if n[0] not in self.forwarded_task_names]
        while len(free_places) > 0 and len(not_stored_yet) > 0:
            self.forwarded_task[free_places[0]] = np.asarray(preprocessed_tasks[not_stored_yet[0]][1:])

            self.forwarded_task_names[free_places[0]] = preprocessed_tasks[not_stored_yet[0]][0]
            free_places = [i for i, n in enumerate(self.forwarded_task_names) if n == str(assumptions.IGNORE)]
            not_stored_yet = [i for i, n in enumerate(preprocessed_tasks) if n[0] not in self.forwarded_task_names]

        if True:
            print("Task List")
            for i in range(len(self.forwarded_task_names)):
                print(f"Task name: {self.forwarded_task_names[i]} \t values: {self.forwarded_task[i]}")


        self.state = np.asarray([self.vision_grid, self.agent_attached, self.forwarded_task, self.energy])

        return self.state