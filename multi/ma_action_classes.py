import ma_assumptions

class Action:
    def __init__(self):
        self.type = ""
        self.param = []


class ActionSkip(Action):
    def __init__(self):
        super(ActionSkip, self).__init__()
        self.type = "skip"
        self.param = [""]

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
for i in range(ma_assumptions.TASK_NUM):
    action_dict[max_key] = ActionSubmit(i)
    max_key += 1
