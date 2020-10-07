from server_env import find_ind_in_observation_np_array

class Action:
    def __init__(self):
        self.type = ""
        self.param = []

class ActionSkip(Action):
    def __init__(self):
        super(ActionSkip, self).__init__()
        self.type = "skip"


class ActionMove(Action):
    """
    Directions:
        - n = North
        - s = South
        - w = West
        - e = East
    """
    def __init__(self, direction: str):
        super(ActionMove, self).__init__()
        self.type = "move"
        self.param = [direction]

    def eval(self, state):
        reward = 0
        lastActionParams, lastAction, lastActionResult, map = state
        up = find_ind_in_observation_np_array(map, (0,-1))
        down = find_ind_in_observation_np_array(map, (0,1))
        right = find_ind_in_observation_np_array(map, (1,0))
        left = find_ind_in_observation_np_array(map, (-1,0))
        move = -1
        if self.param[0] == "n":
            move = up
        elif self.param[0] == "w":
            move = left
        elif self.param[0] == "s":
            move = down
        elif self.param[0] == "e":
            move = right

        print(f"Move {self.param[0]} action eval. Map: {map[move]}, {map[move][2:]}")
        thing, terrain = map[move][2:]
        # Wall move bad
        if terrain == 2:
            reward -= 1


        return reward

