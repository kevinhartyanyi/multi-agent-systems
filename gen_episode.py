from server_env import *
from message_classes import *
from action_classes import *
from random_agent import *

env = Server()

agent_id = 1
agent1 = Random_Agent("agentA1", agent_id, env)

state = env.reset()
agent1.connect()

assert agent1.init_agent() # auth-response
print("YES")
response = agent1.receive() # sim-start (vision, step)
response = agent1.receive() # request-action
print("My first request-action")
actions = []
#step = 0
#max_step = 500

while response["type"] == "request-action":
    agent1.update_env(response)
    action, ind = agent1.act()
    actions.append(ind)
    
    state, reward, done, _ = env.step(ind)
    
    agent1.send(action)
    response = agent1.receive()
    
print(actions)
    