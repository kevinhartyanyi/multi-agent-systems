import itertools
from action_classes import *
import assumptions

action_dict = {
    0: ActionSkip(),
    1: ActionMove("n"),
    2: ActionMove("s"),
    3: ActionMove("e"),
    4: ActionMove("w"),
    5: ActionAttach("n"),
    6: ActionAttach("s"),
    7: ActionAttach("e"),
    8: ActionAttach("w"),
    9: ActionDetach("n"),
    10: ActionDetach("s"),
    11: ActionDetach("e"),
    12: ActionDetach("w"),
    13: ActionRotate("cw"),
    14: ActionRotate("ccw"),
    15: ActionRequest("n"),
    16: ActionRequest("s"),
    17: ActionRequest("e"),
    18: ActionRequest("w")
}
# Add Submit actions
max_key = max(action_dict.keys()) + 1
for i in range(assumptions.TASK_NUM):
    action_dict[max_key] = ActionSubmit(i)
    max_key += 1


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

def calc_reward(perception, task_names, tasks) -> int:
    reward = 0
    success = "success" == perception["lastActionResult"]
    last_action = perception["lastAction"]
    last_action_param = perception["lastActionParams"]
    print(f"Calc Action: {last_action}     Param: {last_action_param}")
    if not success and last_action != "skip":
        print("Reason to fail: ", perception["lastActionResult"])
        reward = -1
    elif last_action == "submit":
        ind = -1
        for i, name in enumerate(task_names):
            if name == last_action_param:
                ind = i
                break
        else:
            print("\n\n\nERROR: Didn't find index for submit (task name)\n\n\n")
        reward = tasks[ind][3]
    return reward
