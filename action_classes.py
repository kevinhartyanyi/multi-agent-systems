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