from enum import Enum
import json

class ServerMessageTypes(Enum):
    AUTHRESPONSE = 1
    SIMSTART = 2
    REQUESTACTION = 3
    SIMEND = 4
    BYE = 5


class AgentMessageTypes(Enum):
    AUTHREQUEST = 1
    ACTION = 2

    def __str__(self):
        re = ""
        if self.value == 1:
            re = "auth-request"
        elif self.value == 2:
            re = "action"
        return re


class AuthRequest:
    def __init__(self, user: str, password: str):
        self.type = "auth-request"
        self.user = user
        self.pw = password
        self.json = {"type": self.type, "content": {}}

    def msg(self) -> str:
        self.json["content"] = {"user": self.user, "pw": self.pw}
        return json.dumps(self.json) + "\0"


class Action:
    def __init__(self, id: int, action_type: str, action_array: list):
        self.type = "action"
        self.id = id
        self.action_type = action_type
        self.action_array = action_array
        self.json = {"type": self.type, "content": {}}

    def msg(self) -> str:
        self.json["content"] = {"id": self.id, "type": self.action_type, "p": self.action_array}
        return json.dumps(self.json) + "\0"
