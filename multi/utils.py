import numpy as np
import random
from collections import deque
import torch

from ma_action_classes import action_dict
#import ma_assumptions

import itertools

class MultiAgentReplayBuffer:

    def __init__(self, num_agents, max_size):
        self.max_size = max_size
        self.num_agents = num_agents
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array(reward), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        obs_batch = [[] for _ in range(self.num_agents)]  # [ [states of agent 1], ... ,[states of agent n] ]    ]
        indiv_action_batch = [[] for _ in range(self.num_agents)] # [ [actions of agent 1], ... , [actions of agent n]]
        indiv_reward_batch = [[] for _ in range(self.num_agents)]
        next_obs_batch = [[] for _ in range(self.num_agents)]

        global_state_batch = []
        global_next_state_batch = []
        global_actions_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)


        for experience in batch:
            state, action, reward, next_state, done = experience

            for i in range(self.num_agents):
                obs_i = state[i]
                action_i = action[i]
                reward_i = reward[i]
                next_obs_i = next_state[i]

                obs_batch[i].append(obs_i)
                indiv_action_batch[i].append(action_i)
                indiv_reward_batch[i].append(reward_i)
                next_obs_batch[i].append(next_obs_i)

            global_state_batch.append(np.concatenate(state))
            global_actions_batch.append(torch.cat(action))
            global_next_state_batch.append(np.concatenate(next_state))
            done_batch.append(done)

        return obs_batch, indiv_action_batch, indiv_reward_batch, next_obs_batch, global_state_batch, global_actions_batch, global_next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)

"""
0. - skip
1. - move[n]
2. - move[s]
3. - move[e]
4. - move[w]
5. - attach[n]
6. - attach[s]
7. - attach[e]
8. - attach[w]
9. - detach[n]
10. - detach[s]
11. - detach[e]
12. - detach[w]
13. - rotate[cw]
14. - rotate[ccw]
15. - request[n]
16. - request[s]
17. - request[e]
18. - request[w]
19 -> 19+task_num. - submit[0-task_num]
"""

_things_dict = {
    0:  {
        'name': 'empty'
        },
    1:  {
        'name': 'entity',
        0: 'A',
        1: 'B'
        },
    2:  {
        'name': 'block',
        0: 'b0',
        1: 'b1',
        2: 'b2'
        },
    3:  {
        'name': 'dispenser' # Duplicate 'block'?
        },
    4:  {
        'name': 'marker',
        0: 'clear',
        1: 'ci',
        2: 'cp'
        }
}

# _things_dict[type]{['code']/['details']}
things_dict = {
    'empty':    {
                'code': 0
                },
    'entity':   {
                'code': 1,
                'details':
                    {
                    'A': 0, # Should we change this (avoid 0), so it's easier to see on the map? A: If its justso we can see it better then I would say no, the network doesnt get the agent map as a parameter
                    'B': 1
                    }
                },
    'block':    {
                'code': 2,
                'details':
                    {
                    'b0': 0, # Should we change this (avoid 0), so it's easier to see on the map?
                    'b1': 1,
                    'b2': 2
                    }
                },
    'dispenser':{
                'code': 3 # Duplicate 'block'?
                },
    'marker':   {
                'code': 4,
                'details':
                    {
                    'clear': 0, # Should we change this (avoid 0), so it's easier to see on the map?
                    'ci': 1,
                    'cp': 2
                    }
                }
}

terrain_dict = {
    'empty': 0,
    'goal': 1,
    'obstacle': 2
}

# TODO Replace 'Should work for deatils and type to' see server_env_new update function for reference and for Dispenser/Block
def get_things_code(name: str):
    return things_dict[name]['code']

def get_things_details(name: str, detail: str):
    return things_dict[name if name != 'dispenser' else 'block']['details'][detail]

def get_terrain_code(name: str):
    return terrain_dict[name]

def vision_grid_size(vision):
    """
    Calculates the maximum visible cell amount based on vision.
    """
    if vision > 0:
        return 4 * vision + vision_grid_size(vision - 1)
    else:
        return 0

def init_vision_grid(vision):
    """
    Returns initialized perception coordinates.
    """
    re = []
    for v in range(-vision, vision + 1):
        re += itertools.product([v], list(range((-vision) + abs(v), (vision + 1) - abs(v))))

    return [list(l) + [0,0,0] for l in re]

def find_coord_index(array, val):
    ind = -1
    for i,v in enumerate(array):
        x,y = v[:2]
        if x == val[0] and y == val[1]:
            ind = i
            break
    return ind

def vision_grid_size(vision):
    """
    Calculates the maximum visible cell amount based on vision.
    """
    if vision > 0:
        return 4 * vision + vision_grid_size(vision - 1)
    else:
        return 0

def get_attached_blocks(things, attached, cords=False):
    blocks = []
    for th in things:
        x = th["x"]
        y = th["y"]
        isAttached = [x == x_a and y == y_a for x_a, y_a in attached]
        #print("isAttached: ", isAttached)
        if any(isAttached):
            detail = th["details"]
            typ = th["type"]
            if cords:
                blocks.append((x,y))
            else:
                blocks.append(get_things_details(typ, detail))
    return blocks

def block_used_in_lastAction(last_action_param, things, cords=False):
    if last_action_param == "e":
        cord = (1,0)
    elif last_action_param == "w":
        cord = (-1,0)
    elif last_action_param == "n":
        cord = (0,-1)
    elif last_action_param == "s":
        cord = (0,1)
    else:
        raise Exception("block_used_in_lastAction, called with wrong last_action_param")

    for th in things:
        x = th["x"]
        y = th["y"]
        typ = th["type"]
        if x == cord[0] and y == cord[1] and typ == "block":
            detail = th["details"]
            if cords:
                return (x, y)
            else:
                return get_things_details(typ, detail)
    return None

def l1_dist(cord1, cord2):
    return abs(cord1[0] - cord2[0]) + abs(cord1[1] - cord2[1])

def get_distance(things_or_terrain, search_typ, attached=None):
    """
    :param things_or_terrain: only use terrain if the search_typ is "goal"
    :param search_typ: "block" or "dispenser" or "goal"
    :return: int
    """
    agent = (0,0)
    dists = []
    if search_typ == "block" or search_typ == "dispenser":
        for th in things_or_terrain:
            x = th["x"]
            y = th["y"]
            typ = th["type"]
            isAttached = [x == x_a and y == y_a for x_a, y_a in attached]
            # Skips the attached blocks
            if typ == search_typ and not any(isAttached):
                dists.append(l1_dist(agent, (x,y)))
    elif search_typ == "goal" and "goal" in things_or_terrain.keys():
        for th in things_or_terrain["goal"]:
            x = th[0]
            y = th[1]
            dists.append(l1_dist(agent, (x,y)))
    #print("Distances: ", dists)
    return min(dists) if len(dists) > 0 else -1

def calc_reward_v2(perception, last_task_names, last_tasks, attached_cords_in_last_response, last_last_action_and_param) -> int:
    reward = 0
    success = "success" == perception["lastActionResult"]
    last_action = perception["lastAction"]
    last_action_param = perception["lastActionParams"][0]
    things = perception["things"]
    terrain = perception["terrain"]
    attached = perception["attached"]

    attached_blocks = get_attached_blocks(things, attached)
    # attached_blocks_cords = get_attached_blocks(things, attached, cords=True)

    blocks_required_by_tasks = [int(i[4]) for i in last_tasks if i[4] != 0.5]

    #print(f"Calc Action: {last_action}     Param: {last_action_param}")
    if last_action == "skip":
        reward = -1


    elif last_action == "move":
        if success:
            #print("Required blocks: ", blocks_required_by_tasks)
            #print("Attached blocks: ", attached_blocks)
            if len(attached_blocks) > 0 and any([item in blocks_required_by_tasks for item in attached_blocks]):
                reward = 2
            elif len(attached_blocks) > 0:
                reward = 1
            else:
                reward = 0
        else:
            reward = -1


    elif last_action == "attach":
        #print("Required blocks: ", blocks_required_by_tasks)
        #print("Block used in last action: ", block_used_in_lastAction(last_action_param, things))
        #print("Last block cord: ", block_cord)
        #print("Attached block cord in last response: ", attached_cords_in_last_response)
        block_cord = block_used_in_lastAction(last_action_param, things, cords=True)
        block_distance = get_distance(things, "block", attached)
        # Skip attaching to already attached block
        if success and any(x == block_cord[0] and block_cord[1] == y for x,y in attached_cords_in_last_response):
            reward = -1
        elif success and any([item == block_used_in_lastAction(last_action_param, things) for item in blocks_required_by_tasks]):
            reward = 20
        elif success:
            reward = 10
        elif block_distance == 1:
            reward = 5
        elif block_distance == 2:
            reward = 3
        elif block_distance == 3:
            reward = 1
        else:
            reward = -1


    elif last_action == "detach":
        if success:
            reward = 10
        elif len(attached_blocks) > 0:
            reward = 1
        else:
            reward = -1


    elif last_action == "rotate":
        if len(attached_blocks) == 0:
            reward = -1
        else:
            reward = 0


    elif last_action == "request":
        dispenser_distance = get_distance(things, "dispenser", attached)
        if success:
            reward = 10
        elif dispenser_distance == 1:
            reward = 5
        elif dispenser_distance == 2:
            reward = 3
        elif dispenser_distance == 3:
            reward = 1
        else:
            reward = -1

    elif last_action == "submit":
        goal_distance = get_distance(terrain, "goal", attached)
        if success:
            #print("Success submiting the task")
            ind = -1
            for i, name in enumerate(last_task_names):
                if name == last_action_param:
                    ind = i
                    break
            else:
                print("\n\n\nERROR: Didn't find index for submit (task name)\n\n\n")
            reward = 500 + last_tasks[ind][3]
        elif last_last_action_and_param[0] == last_action and last_last_action_and_param[1] == last_action_param:
            reward = -1
        elif goal_distance == 0 and len(attached_blocks) > 0:
            reward = 60
        elif goal_distance == 1 and len(attached_blocks) > 0:
            reward = 50
        elif goal_distance == 2 and len(attached_blocks) > 0:
            reward = 40
        elif goal_distance == 3 and len(attached_blocks) > 0:
            reward = 30
        elif goal_distance == 0 and len(attached_blocks) == 0:
            reward = 20
        elif goal_distance == 1 and len(attached_blocks) == 0:
            reward = 10
        elif goal_distance == 2 and len(attached_blocks) == 0:
            reward = 5
        elif goal_distance == 3 and len(attached_blocks) == 0:
            reward = 3
        else:
            reward = -1
    return reward
