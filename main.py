from server import *
from message_classes import *
from action_classes import *
import random

directions = ["n", "s", "w", "e"]

agent_id = 1
agent1 = AuthRequest("agentA1", agent_id)

server = Server()

success = server.init_agent(agent1)
print(f"Init successful: {success}")

response = server.receive()
while response["type"] != "bye":

    if response["type"] == "sim-start":
        print("Do something")  # Probably nothing
    elif response["type"] == "sim-end":
        content = response["content"]
        print("End of simulation")
        print(f"Results: \n\t"
              f"Score: {content['score']}\n\t"
              f"Ranking: {content['ranking']}\n\t"
              f"Time: {content['time']}")
    elif response["type"] == "request-action":
        content = response["content"]
        print(content["percept"])


        print("Send action")
        high_level_thinking = directions[random.randint(0, len(directions) - 1)]
        action = ActionMove(high_level_thinking)
        server.send(ActionReply(content["id"], action))

    response = server.receive()

print("end")
