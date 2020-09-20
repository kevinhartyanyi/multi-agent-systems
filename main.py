from server import *
from message_classes import *

agent1 = AuthRequest("agentA1", "1")

server = Server()

success = server.init_agent(agent1)
print(f"Init successful: {success}")

response = server.receive()
while response["type"] != "bye":

    if response["type"] == "sim-start":
        print("Do something")
    elif response["type"] == "request-action":
        print("Send action")
        # server.send()

    response = server.receive()

print("end")