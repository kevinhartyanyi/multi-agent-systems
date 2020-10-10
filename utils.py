import itertools
from action_classes import *

action_dict = {
    0: ActionSkip(),
    1: ActionMove("n"),
    2: ActionMove("s"),
    3: ActionMove("e"),
    4: ActionMove("w")
}
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
15. - submit[0-task_num]
"""

things_dict = {
    0:  {
        'name': 'empty'
        },
    1:  {
        'name': 'entity',
        0: 'A',
        1: 'B'
        }
}

terrain_dict = {
    0: 'empty',
    1: 'goal',
    2: 'obstacle'
}

# TODO Replace 'Should work for deatils and type to' see server_env_new update function for reference and for Dispenser/Block
def get_thing_num(name: str):
    return 1 if name == "entity" else 2

def get_terrain_num(name: str): # TODO Replace
    return 1 if name == "goal" else 2
    

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
    print(f"Last Action: {last_action}     Param: {last_action_param}")
    if not success:
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

"""{'lastActionParams': ['w'], 'score': 0, 'lastAction': 'move', 'things': [{'x': 0, 'y': 0, 'details': 'A', 
'type': 'entity'}], 'attached': [], 'disabled': False, 'terrain': {'obstacle': [[3, 2], [2, 2], [1, 2], [0, 2], [-1, 
2], [-2, 2], [-5, 0], [-3, 2], [1, -4], [0, -4], [-1, -4]]}, 'lastActionResult': 'success', 'tasks': [{'reward': 10, 
'requirements': [{'x': 0, 'y': 1, 'details': '', 'type': 'b2'}], 'name': 'task11', 'deadline': 408}, """