import itertools
from action_classes import *

action_dict = {
    0: ActionSkip()
}

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