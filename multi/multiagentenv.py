import numpy as np
from subprocess import Popen, PIPE

import ma_assumptions
from utils import * #vision_grid_size

class MultiAgentEnv():
    def __init__(self):
        self.agent_vision = ma_assumptions.VISION_RANGE
        self.n = ma_assumptions.NUM_AGENTS

        self.vision_grid = []
        self.agent_attached = []
        self.forwarded_task_names = []
        self.forwarded_task = []
        self.energy = []
        self.disabled = []
        self.lastActionResult = []
        self.state = []

        for i in range(self.n):
            # Current perception
            self.vision_grid.append(ma_assumptions.IGNORE * np.ones((vision_grid_size(self.agent_vision)+1, 5))) # Things, terrain
            self.agent_attached.append(ma_assumptions.IGNORE * np.ones((vision_grid_size(ma_assumptions.TASK_SIZE), 2))) # Attached -> Extract attached type from lastAction + lastActionParameter
            self.forwarded_task_names.append([str(ma_assumptions.IGNORE)] * ma_assumptions.TASK_NUM) # Names of the tracked tasks
            self.forwarded_task.append(ma_assumptions.IGNORE * np.ones((ma_assumptions.TASK_NUM, (2 + ma_assumptions.TASK_SIZE * 3)))) # x, y, deadline, points, block_num

            self.energy.append(np.array([ma_assumptions.IGNORE]))

            self.disabled.append(False)
            self.lastActionResult.append('success')

            self.state.append(None)

    def reset(self, monitor=False):
        self.__init__()
        if monitor:
            process = Popen(
                ["java", "-jar", "massim-2019-2.0/server/server-2019-2.1-jar-with-dependencies.jar", "--monitor", "8000",
                 "-conf", "massim-2019-2.0/server/conf/SampleConfig-Deliverable1.json"],
                stdout=PIPE, stderr=PIPE, stdin=PIPE)
        else:
            process = Popen(
                ["java", "-jar", "massim-2019-2.0/server/server-2019-2.1-jar-with-dependencies.jar",
                 "-conf", "massim-2019-2.0/server/conf/SampleConfig-Deliverable1.json"],
                stdout=PIPE, stderr=PIPE, stdin=PIPE)

    def observation_space(self, agent_id):
        return self.vision_grid[agent_id].size + self.agent_attached[agent_id].size + self.forwarded_task[agent_id].size + self.energy[agent_id].size

    def get_task_name(self, agent_id, index):
        return self.forwarded_task_names[agent_id][index]

    def step(self, actions: int):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}

        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])
        # advance world state
        self.world.step()
        # record observation for each agent
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            reward_n.append(self._get_reward(agent))
            done_n.append(self._get_done(agent))

            info_n['n'].append(self._get_info(agent))

        # all agents get total reward in cooperative case
        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [reward] * self.n

        return obs_n, reward_n, done_n, info_n

    def get_state_size(self, agent_id=0):
        return self.vision_grid[agent_id].size + self.agent_attached[agent_id].size + self.forwarded_task[agent_id].size + self.energy[agent_id].size

    def update(self, msg, agent_id):
        # Things and Terrain
        observation_map = init_vision_grid(self.agent_vision)
        things = msg['things']
        terrain = msg['terrain']
        # print(f"\n\n\nThings: {things}")
        # print(f"Terrain: {terrain}")
        self.energy[agent_id][0] = msg['energy']
        attached = msg['attached'] # List of coordinates

        if len(attached) > 0:

            # Update agent_attached
            attached = np.asarray(attached)
            size_diff = self.agent_attached[agent_id].shape[0] - attached.shape[0]
            if size_diff > 0:
                self.agent_attached[agent_id] = np.vstack([attached, ma_assumptions.IGNORE * np.ones((size_diff, 2)) ])
            else:
                self.agent_attached[agent_id] = attached[:self.agent_attached[agent_id].shape[0], :] # Just a precaution, in case agent_attached isnt large enough

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


        self.vision_grid[agent_id] = np.asarray(observation_map)



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
        for i, name in enumerate(self.forwarded_task_names[agent_id]):
            if name not in task_names and name != str(ma_assumptions.IGNORE):  # Delete if task is over
                self.forwarded_task[agent_id][i] = ma_assumptions.IGNORE * np.ones(2 + ma_assumptions.TASK_SIZE * 3)
            elif name in task_names:  # Update otherwise
                for t in preprocessed_tasks:
                    if t[0] == name:
                        self.forwarded_task[agent_id][i] = np.asarray(t[1:])
                        break


        free_places = [i for i, n in enumerate(self.forwarded_task_names[agent_id]) if n == str(ma_assumptions.IGNORE)]
        not_stored_yet = [i for i, n in enumerate(preprocessed_tasks) if n[0] not in self.forwarded_task_names[agent_id]]
        while len(free_places) > 0 and len(not_stored_yet) > 0:
            self.forwarded_task[agent_id][free_places[0]] = np.asarray(preprocessed_tasks[not_stored_yet[0]][1:])

            self.forwarded_task_names[agent_id][free_places[0]] = preprocessed_tasks[not_stored_yet[0]][0]
            free_places = [i for i, n in enumerate(self.forwarded_task_names[agent_id]) if n == str(ma_assumptions.IGNORE)]
            not_stored_yet = [i for i, n in enumerate(preprocessed_tasks) if n[0] not in self.forwarded_task_names[agent_id]]

        if False:
            print("Task List")
            for i in range(len(self.forwarded_task_names[agent_id])):
                print(f"Task name: {self.forwarded_task_names[agent_id][i]} \t values: {self.forwarded_task[agent_id][i]}")


        self.state[agent_id] = np.asarray([self.vision_grid[agent_id], self.agent_attached[agent_id], self.forwarded_task[agent_id], self.energy[agent_id]])

        return self.state[agent_id]

""" To be implemented:
    env.observation_space[agent_id].shape[0]
    env.action_space[agent_id].n
    env.n
    env.reset()
    env.step(actions) - next_states, rewards, dones, _
"""
