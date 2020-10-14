from server_env_new import *
from random_agent import *
from reinforce_agent import *
from dqn_network import *
from subprocess import Popen, PIPE
import matplotlib.pyplot as plt

def plot_rewards(rewards, name):
    plt.plot(rewards)
    plt.title('Training Avg Rewards')
    plt.xlabel('Episode number')
    plt.ylabel('Average Reward')
    plt.savefig(f"Rewards_{name}.png")

BATCH_SIZE = 5
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

n_actions = len(action_dict)

policy_net = DQN(25, 24, n_actions).to(device).to(float)
target_net = DQN(25, 24, n_actions).to(device).to(float)

policy_net.load_state_dict(torch.load("policy_net_best.pth"))
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()


optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)
steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

episode_rewards = []

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device).to(float)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


env = Server()
num_episodes = 1

agent_id = 1

EXTRA_SMART = True # CHANGE AT OWN RISK
agent1 = None

if EXTRA_SMART:
    agent1 = Reinforce_Agent("agentA1", agent_id, env)
else:
    agent1 = Random_Agent("agentA1", agent_id, env)


monitor = False

for i_episode in range(num_episodes):
    # Initialize the environment and state
    state = env.reset()

    # Start server
    if monitor:
        process = Popen(
            ["java", "-jar", "massim-2019-2.0/server/server-2019-2.1-jar-with-dependencies.jar", "--monitor", "8000",
             "-conf", "massim-2019-2.0/server/conf/SampleConfig-Deliverable1.json"],
            stdout=PIPE, stderr=PIPE, stdin=PIPE)
    else:
        process = Popen(
            ["java", "-jar", "massim-2019-2.0/server/server-2019-2.1-jar-with-dependencies.jar",
             "-conf", "massim-2019-2.0/server/conf/SampleConfig-Deliverable1.json"],
            stdout=PIPE, stderr=PIPE, stdin=PIPE)

    time.sleep(5)
    agent1.connect()
    assert agent1.init_agent()  # auth-response
    print("YES")
    time.sleep(2)
    process.stdin.write(b'\n')
    process.stdin.flush()

    response = agent1.receive()  # sim-start (vision, step)
    response = agent1.receive()  # request-action
    print("My first request-action")

    agent1.update_env(response)
    state = torch.from_numpy(agent1.get_state()).unsqueeze(0).unsqueeze(0)
    #xstate = state.double()

    collect_rewards = []

    for t in count():
        # Select and perform an action


        action = select_action(state)
        print("Selected action: ", action)
        if isinstance(action_dict[action.item()],
            ActionSubmit):  # TODO Could be performance improved by using max_key in utils
            action_dict[action.item()].init_task_name(env.forwarded_task_names)

        agent1.send(action.item())
        response = agent1.receive()

        done = response["type"] != "request-action"



        #_, reward, done, _ = env.step(action.item())


        # Observe new state
        if not done:
            rew = calc_reward(response['content']['percept'], env.forwarded_task_names, env.forwarded_task)
            collect_rewards.append(rew)
            reward = torch.tensor([[rew]])

            agent1.update_env(response)
            next_state = torch.from_numpy(agent1.get_state()).type(torch.DoubleTensor).unsqueeze(0).unsqueeze(0)
        else:
            collect_rewards.append(0)
            reward = torch.tensor([[0]])
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            episode_rewards.append(np.average(collect_rewards))
            break

    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
        print("Target Update")

    if i_episode % 10:
        torch.save(policy_net.state_dict(), f"policy_net_{i_episode}.pth")
        torch.save(target_net.state_dict(), f"target_net_{i_episode}.pth")
        plot_rewards(episode_rewards, num_episodes)

    process.kill()
    agent1.reset()

print('Complete')
plot_rewards(episode_rewards, "best")

torch.save(policy_net.state_dict(), f"policy_net_best.pth")
torch.save(target_net.state_dict(), f"target_net_best.pth")