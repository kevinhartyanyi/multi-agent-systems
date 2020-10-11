from server_env import find_ind_in_observation_np_array # Change to utils.find_coord_index REMOVE THIS


class Action:
    def __init__(self):
        self.type = ""
        self.param = []


class ActionSkip(Action):
    def __init__(self):
        super(ActionSkip, self).__init__()
        self.type = "skip"

    def print(self, reward):
        print(f"\n\nCurrent action: \n\tSkip \t Reward: {reward}")


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

    def print(self, reward):
        print(f"\n\nCurrent action: \n\tMove: {self.param[0]} \t Reward: {reward}")

    def eval(self, state):
        reward = 0
        lastActionParams, lastAction, lastActionResult, map = state
        up = find_ind_in_observation_np_array(map, (0, -1))
        down = find_ind_in_observation_np_array(map, (0, 1))
        right = find_ind_in_observation_np_array(map, (1, 0))
        left = find_ind_in_observation_np_array(map, (-1, 0))
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


class ActionAttach(Action):
    """
    Directions:
        - n = North
        - s = South
        - w = West
        - e = East
    """

    def __init__(self, direction: str):
        super(ActionAttach, self).__init__()
        self.type = "attach"
        self.param = [direction]

    def print(self, reward):
        print(f"\n\nCurrent action: \n\tAttach: {self.param[0]} \t Reward: {reward}")


class ActionDetach(Action):
    """
    Directions:
        - n = North
        - s = South
        - w = West
        - e = East
    """

    def __init__(self, direction: str):
        super(ActionDetach, self).__init__()
        self.type = "detach"
        self.param = [direction]

    def print(self, reward):
        print(f"\n\nCurrent action: \n\tDetach: {self.param[0]} \t Reward: {reward}")


class ActionRotate(Action):
    """
    Rotation:
        - cw  = clockwise
        - ccw = counterclockwise
    """

    def __init__(self, rotation: str):
        super(ActionRotate, self).__init__()
        self.type = "rotate"
        self.param = [rotation]

    def print(self, reward):
        print(f"\n\nCurrent action: \n\tRotate: {self.param[0]} \t Reward: {reward}")


class ActionRequest(Action):
    """
    Directions:
        - n = North
        - s = South
        - w = West
        - e = East
    """

    def __init__(self, direction: str):
        super(ActionRequest, self).__init__()
        self.type = "request"
        self.param = [direction]

    def print(self, reward):
        print(f"\n\nCurrent action: \n\tRequest: {self.param[0]} \t Reward: {reward}")

class ActionSubmit(Action):
    """
    Directions:
        - n = North
        - s = South
        - w = West
        - e = East
    """

    def __init__(self, task_num: int):
        super(ActionSubmit, self).__init__()
        self.type = "submit"
        self.param = []
        self.task_num = task_num

    def init_task_name(self, task_names):
        self.param = [task_names[self.task_num]]

    def print(self, reward):
        print(f"\n\nCurrent action: \n\tSubmit: {self.param[0]} \t Reward: {reward}")
