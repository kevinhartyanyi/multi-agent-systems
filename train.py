from server_env_new import *
from random_agent import *
from reinforce_agent import *
from dqn_network import *
from subprocess import Popen, PIPE
import matplotlib.pyplot as pltbiztos

def plot_rewards(rewards, name):
    plt.clf()
    plt.plot(rewards)
    plt.title('Training Avg Rewards')
    plt.xlabel('Episode number')
    plt.ylabel('Average Reward')
    plt.savefig(f"plots/Rewards_reward2_{name}.png")

def plot_actions(actions, name):
    plt.clf()
    fig, ax = plt.subplots(figsize=(20,10))
    n, bins, patches = ax.hist(actions, len(action_dict))
    ax.set_xlabel('Actions')
    ax.set_ylabel('Number of times chosen by the agent')
    ax.set_title('Actions Histogram')
    ax.set_xticks(actions)
    plt.savefig(f"plots/Actions_histogram_reward2_{name}.png")

def plot_double_action(actions, name):

    failed = [len(list(filter(lambda y: y < 0, v))) for k, v in actions.items()]
    correct = [len(list(filter(lambda y: y >= 0, v))) for k, v in actions.items()]


    plt.clf()
    fig, ax = plt.subplots(figsize=(20, 10))
    #n, bins, patches = ax.hist([failed, correct], list(range(len(action_dict))), density=True, histtype='bar', stacked=True)

    N = len(action_dict)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bars: can also be len(x) sequence

    p1 = plt.bar(ind, correct, width)
    p2 = plt.bar(ind, failed, width)

    ax.set_xlabel('Actions')
    ax.set_ylabel('Number of times chosen by the agent')
    ax.set_title('Actions Histogram')
    ax.set_xticks(list(range(len(action_dict))))
    plt.legend((p1[0], p2[0]), ('Correct', 'Failed'))
    #plt.show()
    plt.savefig(f"plots/Actions_histogram_reward_{name}.png")


env = Server()
num_episodes = 1000

agent_id = 1

EXTRA_SMART = True # CHANGE AT OWN RISK
agent1 = None

if EXTRA_SMART:
    agent1 = Reinforce_Agent("agentA1", agent_id, env)
else:
    agent1 = Random_Agent("agentA1", agent_id, env)



BATCH_SIZE = 1
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

n_actions = len(action_dict)

model_conv = False

if model_conv:
    policy_net = DQN(25, 24, n_actions).to(device).to(float)
    target_net = DQN(25, 24, n_actions).to(device).to(float)
else:
    input_size = agent1.get_input_size()
    policy_net = FFNet(input_size, n_actions).to(device).to(float)
    target_net = FFNet(input_size, n_actions).to(device).to(float)


#policy_net.load_state_dict(torch.load("weights/policy_net_best.pth"))
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
            #return policy_net(state).max(1)[1].view(1, 1)
            return policy_net(state).argmax()
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

episode_rewards = []
selected_actions = []
selected_action_dict = {}
for act in action_dict.keys():
    selected_action_dict[act] = []

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))
    #print(batch)

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    #print(batch.next_state)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    #print("Batch action: ", batch.action)
    #print("Batch action shape: ", batch.action[0].dim)
    #print("Batch squeezed action shape: ", batch.action[0].unsqueeze(0).shape)
    batch_action_0 = batch.action[0]
    #print("Len: ", len(batch.action))
    while (len(batch_action_0.shape) < 2):  # Hmmm
        #print("Increase")
        batch_action_0 = batch_action_0.unsqueeze(0)
    batch = batch._replace(action=(batch_action_0,))
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    """
    state_action_values_tmp:  tensor([[ 0.6033, -0.0280, -0.6558,  0.0978,  0.2147,  0.1978,  0.5327, -0.0957,
          0.1902, -0.6467,  0.2421,  0.4637,  0.3302,  0.3465,  0.0197,  0.3544,
         -0.9784,  0.0170, -0.6998, -0.4992,  0.0021, -0.4202,  0.4588, -0.0674]],
       dtype=torch.float64, grad_fn=<AddmmBackward>)
    state_action_values_tmp.shape:  torch.Size([1, 24])
    state_action_values: tensor([[0.0978]], dtype=torch.float64, grad_fn=<GatherBackward>)
    state_action_values shape: torch.Size([1, 1])
    Press something/home/kevin/Programming/School/MSc_1/multi_agent/new_model/multi-agent-systems/train.py:157: UserWarning: Using a target size (torch.Size([1, 1, 1])) that is different to the input size (torch.Size([1, 1])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.
      loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    """
    state_action_values_tmp = policy_net(state_batch)
    #print("Base state_action_values_tmp: ", state_action_values_tmp)
    #print("Base state_action_values_tmp.shape: ", state_action_values_tmp.shape)

    state_action_values_ff = policy_net(state_batch).unsqueeze(0)
    #print("FF state_action_values_ff: ", state_action_values_ff)
    #print("FF state_action_values_ff.shape: ", state_action_values_ff.shape)

    if model_conv:
        state_action_values = policy_net(state_batch).gather(1, action_batch)
    else:
        state_action_values = policy_net(state_batch).unsqueeze(0)
        #print("FF dims:", len(state_action_values.shape))
        while(len(state_action_values.shape) > 2): # Hmmm
            #print("Reduce")
            state_action_values = state_action_values.squeeze(0)
        #print("FF dims after:", len(state_action_values.shape))
        state_action_values = state_action_values.gather(1, action_batch)
    #
    #print("USED state_action_values:",state_action_values)
    #print("USED state_action_values shape:",state_action_values.shape)


    # state_action_values: tensor([[-0.2770]], dtype=torch.float64, grad_fn= < GatherBackward >)
    # shape: torch.Size([1, 1])

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device).to(float)
    if model_conv:
        target = target_net(non_final_next_states)
    else:
        target = target_net(non_final_next_states).squeeze(0)
    #print("Target: ", target)
    #print("Target shape: ", target.shape)
    #print("Target Net result: ", target.max(1)[0].detach())
    #print("Values:",next_state_values)
    #print("Values shape:",next_state_values.shape)
    #print("Non final:",non_final_mask)
    #print("Non final shape :",non_final_mask.shape)
    #print("Next final:",next_state_values[non_final_mask])
    #print("Next final shape:",next_state_values[non_final_mask].shape)
    next_state_values[non_final_mask] = target.max(1)[0].detach()
    """    
    Target:  tensor([[-1.6124, -0.9181, -0.2592, -0.0746,  1.9099, -0.1863, -0.7488, -0.7087,
              1.4476, -0.4261, -0.3391, -0.6141,  1.3410,  0.7198,  0.9932, -0.2391,
             -2.5868, -0.0864,  0.3222, -0.4475,  0.0884, -0.4845,  0.1519,  0.5026]],
           dtype=torch.float64, grad_fn=<AddmmBackward>)
    Target shape:  torch.Size([1, 24])
    Target Net result:  tensor([1.9099], dtype=torch.float64)
    Values: tensor([0.], dtype=torch.float64)
    Values shape: torch.Size([1])
    Non final: tensor([True])
    Non final shape : torch.Size([1])
    Next final: tensor([0.], dtype=torch.float64)
    Next final shape: torch.Size([1])
    """
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    if model_conv:
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    else:
        """print("loss state_action_values:", state_action_values)
        print("loss state_action_values:", state_action_values.shape)
        print("loss expected_state_action_values:", expected_state_action_values)
        print("loss expected_state_action_values:", expected_state_action_values.shape)"""
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    #print(list(policy_net.parameters()))
    #print("param len:",len(list(policy_net.parameters())))
    for param in policy_net.parameters():
        #print(param.shape)
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


monitor = True

for i_episode in range(num_episodes):
    print("Episode: ", i_episode)
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
    #print("YES")
    time.sleep(2)
    process.stdin.write(b'\n')
    process.stdin.flush()

    response = agent1.receive()  # sim-start (vision, step)
    response = agent1.receive()  # request-action
    #print("My first request-action")

    agent1.update_env(response)
    if model_conv:
        state_conv = torch.from_numpy(agent1.get_state()).unsqueeze(0).unsqueeze(0)
        state = state_conv
    else:
        state_ff = torch.from_numpy(agent1.get_state())
        state = state_ff
    #print("State shape:", state.shape) # State shape: torch.Size([1, 1, 25, 24])
    #xstate = state.double()

    attached_cords_in_last_response = [] # For the calc_reward_v2 function so it won't give points if the agent attaches to an already attached block
    last_lastAction = None # Best name EUNE (for the calc_reward_v2 function task rewards)
    last_lastAction_param = None # Best name EUW
    last_task_names = []
    last_tasks = []

    collect_rewards = []

    for t in count():
        # Select and perform an action


        action = select_action(state)


        #print("Selected action (agent): ", action)

        #action = torch.tensor([[int(input("Action:"))]], device=device, dtype=torch.long)




        if isinstance(action_dict[action.item()],
            ActionSubmit):  # TODO Could be performance improved by using max_key in utils
            action_dict[action.item()].init_task_name(env.forwarded_task_names)

        agent1.send(action.item())
        response = agent1.receive()

        done = response["type"] != "request-action"



        #_, reward, done, _ = env.step(action.item())


        # Observe new state
        if not done:
            last_last_action_and_param = (last_lastAction, last_lastAction_param)
            rew = calc_reward_v2(response['content']['percept'], last_task_names, last_tasks, attached_cords_in_last_response, last_last_action_and_param)
            attached_cords_in_last_response = get_attached_blocks(response['content']['percept']['things'],
                                                                  response['content']['percept']['attached'], cords=True)
            last_lastAction = response['content']['percept']['lastAction']
            last_lastAction_param =  response['content']['percept']['lastActionParams'][0]
            last_task_names = env.forwarded_task_names
            last_tasks = env.forwarded_task

            collect_rewards.append(rew)
            selected_actions.append(action.item())
            selected_action_dict[action.item()].append(rew)
            reward = torch.tensor([[rew]])

            agent1.update_env(response)
            next_state = torch.from_numpy(agent1.get_state()).type(torch.DoubleTensor).unsqueeze(0).unsqueeze(0)
        else:
            collect_rewards.append(0)
            reward = torch.tensor([[0]])
            next_state = None

        #print("Agent Reward:", reward)
        #action_dict[action.item()].print(reward.item())
        #print("\n")

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            episode_rewards.append(np.average(collect_rewards))
            break

        #input("Press something")

    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
        #print("Target Update")

    if i_episode % 1 == 0:
        torch.save(policy_net.state_dict(), f"weights/policy_net_{i_episode}.pth")
        torch.save(target_net.state_dict(), f"weights/target_net_{i_episode}.pth")
        plot_rewards(episode_rewards, i_episode)
        plot_double_action(selected_action_dict, i_episode)

    process.kill()
    agent1.reset()

print('Complete')
plot_rewards(episode_rewards, "best")
plot_double_action(selected_action_dict, "best")

torch.save(policy_net.state_dict(), f"weights/policy_net_best.pth")
torch.save(target_net.state_dict(), f"weights/target_net_best.pth")
