from server_env_new import *
from random_agent import *
from reinforce_agent import *

"""
TODO: Check if requires change according to:
https://github.com/openai/gym/blob/master/docs/creating-environments.md
https://stackoverflow.com/questions/45068568/how-to-create-a-new-gym-environment-in-openai
"""
# env = gym.make(Server())
env = Server()

agent_id = 1

EXTRA_SMART = False # CHANGE AT OWN RISK
agent1 = None

if EXTRA_SMART:
    agent1 = Reinforce_Agent("agentA1", agent_id, env)
else:
    agent1 = Random_Agent("agentA1", agent_id, env)

state = env.reset()
agent1.connect()

assert agent1.init_agent()  # auth-response
print("YES")
response = agent1.receive()  # sim-start (vision, step)
response = agent1.receive()  # request-action
print("My first request-action")
actions = []
rewards = []
log_probs = []

while response["type"] == "request-action":
    agent1.update_env(response)
    
    if EXTRA_SMART:
        action_ind, log_prob = agent1.act()
        print("ACTION:", action_ind)
        log_probs.append(log_prob)
    else:
        action_ind = agent1.act()
    actions.append(action_ind)

    # state, reward, done, _ = env.step(action_ind)

    if isinstance(action_dict[action_ind], ActionSubmit):  # TODO Could be performance improved by using max_key in utils
        action_dict[action_ind].init_task_name(env.forwarded_task_names)

    agent1.send(action_ind)
    response = agent1.receive()

    if response["type"] == "request-action":  # We don't get reward for the last action :(
        reward = calc_reward(response['content']['percept'], env.forwarded_task_names, env.forwarded_task)
        rewards.append(reward)
        action_dict[action_ind].print(reward)
        if EXTRA_SMART:
            agent1.update_net(rewards, log_probs)

    # TODO: Only for testing
    # time.sleep(10)
    input()

print(actions)
