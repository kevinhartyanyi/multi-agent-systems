from server_env_new import Server
from random_agent import Random_Agent
from reinforce_agent import Reinforce_Agent
import random
import time

# env = gym.make(Server())
env = Server()

agent_pass = 10

EXTRA_SMART = False # CHANGE AT OWN RISK
agent0 = None
agent1 = None

if EXTRA_SMART:
    agent0 = Reinforce_Agent("agentA1", agent_id, env)
else:
    agent0 = Random_Agent("agentA0", agent_pass, env)
    agent1 = Random_Agent("agentA1", agent_pass, env)

#state = env.reset()
agent0.connect()
assert agent0.init_agent()  # auth-response
response = agent0.receive()  # request-action

agent1.connect()
assert agent1.init_agent()  # auth-response
response = agent1.receive()  # sim-start (vision, step)
print("YES")

print('0:')



print('1:')
response1 = agent0.receive()  # sim-start (vision, step)
response = agent1.receive()  # request-action

agent0.update_env(response1)
agent1.update_env(response)

print("My first request-action")
actions = []
rewards = []
log_probs = []
i = 0
while response["type"] == "request-action":
    i += 1
    #agent1.update_env(response)

    if EXTRA_SMART:
        action_ind, log_prob = agent1.act()
        print("ACTION:", action_ind)
        log_probs.append(log_prob)
    else:
        #action_ind = agent1.act()
        action_ind_0 = random.randint(1, 4)
        action_ind_1 = random.randint(1, 4)
    print("0:", action_ind_0)
    print("1:", action_ind_0)
    #actions.append(action_ind)

    # state, reward, done, _ = env.step(action_ind)

    #if isinstance(action_dict[action_ind], ActionSubmit):  # TODO Could be performance improved by using max_key in utils
    #    action_dict[action_ind].init_task_name(env.forwarded_task_names)

    agent0.send(action_ind_0)
    agent1.send(action_ind_1)

    print('0:')

    response = agent0.receive()
    agent0.update_env(response)

    print('1:')

    response = agent1.receive()
    agent1.update_env(response)

    """
    if response["type"] == "request-action":  # We don't get reward for the last action :(
        reward = calc_reward(response['content']['percept'], env.forwarded_task_names, env.forwarded_task)
        rewards.append(reward)
        action_dict[action_ind].print(reward)
        if EXTRA_SMART and i%4==0:
            agent1.update_net(rewards, log_probs, i > 4)

    """



    # TODO: Only for testing
    #time.sleep(0.5)
    #input()



print(actions)
