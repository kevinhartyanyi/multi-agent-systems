from server_env import *
from message_classes import *
from action_classes import *
from random_agent import *

"""
TODO: Check if requires change according to:
https://github.com/openai/gym/blob/master/docs/creating-environments.md
https://stackoverflow.com/questions/45068568/how-to-create-a-new-gym-environment-in-openai
"""
env = gym.make(Server())

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
    