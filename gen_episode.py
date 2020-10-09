from server_env_new import *
from random_agent import *

"""
TODO: Check if requires change according to:
https://github.com/openai/gym/blob/master/docs/creating-environments.md
https://stackoverflow.com/questions/45068568/how-to-create-a-new-gym-environment-in-openai
"""
#env = gym.make(Server())
env = Server()

agent_id = 1
agent1 = Random_Agent("agentA1", agent_id, env)

state = env.reset()
agent1.connect()

assert agent1.init_agent() # auth-response
print("YES")
response, _ = agent1.receive() # sim-start (vision, step)
response, _ = agent1.receive() # request-action
print("My first request-action")
actions = []
#step = 0
#max_step = 500

while response["type"] == "request-action":
    agent1.update_env(response)
    action_ind = agent1.act()
    actions.append(action_ind)
    
    #state, reward, done, _ = env.step(action_ind)
    
    agent1.send(action_ind)
    response, reward = agent1.receive()  # We don't get reward for the last action :(

    action_dict[action_ind].print(reward)


    # TODO: Only for testing
    #time.sleep(10)
    input()
    
print(actions)
    