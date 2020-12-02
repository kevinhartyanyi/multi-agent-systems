import torch
import numpy as np

from agent import DDPGAgent
from utils import MultiAgentReplayBuffer

import time
import matplotlib.pyplot as plt

from ma_action_classes import action_dict, ActionSubmit
from utils import calc_reward_v2, get_attached_blocks
from log import Log

class MADDPG:

    def __init__(self, env, buffer_maxlen, run_name = "test"):
        self.env = env
        self.num_agents = env.n
        self.replay_buffer = MultiAgentReplayBuffer(self.num_agents, buffer_maxlen)
        self.agents = [DDPGAgent(self.env, i) for i in range(self.num_agents)]
        self.log = Log(name=run_name)

    def get_actions(self, states):
        actions = []
        for i in range(self.num_agents):
            action = self.agents[i].get_action(states[i])
            actions.append(action)
        return actions

    def update(self, batch_size):
        obs_batch, indiv_action_batch, indiv_reward_batch, next_obs_batch, \
            global_state_batch, global_actions_batch, global_next_state_batch, done_batch = self.replay_buffer.sample(batch_size)

        for i in range(self.num_agents):
            obs_batch_i = obs_batch[i]
            indiv_action_batch_i = indiv_action_batch[i]
            indiv_reward_batch_i = indiv_reward_batch[i]
            next_obs_batch_i = next_obs_batch[i]

            next_global_actions = []
            for agent in self.agents:
                next_obs_batch_i = torch.FloatTensor(next_obs_batch_i)
                indiv_next_action = agent.actor.forward(next_obs_batch_i)
                indiv_next_action = [agent.onehot_from_logits(indiv_next_action_j) for indiv_next_action_j in indiv_next_action]
                indiv_next_action = torch.stack(indiv_next_action)
                next_global_actions.append(indiv_next_action)
            next_global_actions = torch.cat([next_actions_i for next_actions_i in next_global_actions], 1)

            self.agents[i].update(indiv_reward_batch_i, obs_batch_i, global_state_batch, global_actions_batch, global_next_state_batch, next_global_actions)
            self.agents[i].target_update()

    def run(self, max_episode, max_steps, batch_size, monitor=False):
        episode_rewards = []
        selected_actions = []
        selected_action_dict = {}
        for act in action_dict.keys():
            selected_action_dict[act] = []

        for episode in range(max_episode):
            attached_cords_in_last_response = [] # For the calc_reward_v2 function so it won't give points if the agent attaches to an already attached block
            last_lastAction = [] # Best name EUNE (for the calc_reward_v2 function task rewards)
            last_lastAction_param = [] # Best name EUW
            last_task_names = []
            last_tasks = []
            responses = []
            states = []

            self.env.reset(monitor=monitor)
            for agent in self.agents:
                _, _ = agent.reset()
            for agent in self.agents:
                response = agent.receive()
                responses.append(response)
                #print(response)
                agent.update_env(response)
                attached_cords_in_last_response.append([])
                last_lastAction.append(None)
                last_lastAction_param.append(None)
                last_task_names.append([])
                last_tasks.append([])
                states.append(agent.get_state())
            time.sleep(2) # Wait to initialize

            #self.env.start_server()
            episode_reward = 0

            for step in range(max_steps):
                actions = self.get_actions(states)
                # Send all actions
                for i,action in enumerate(actions):
                    cpu_action = action.cpu().numpy()
                    action = np.where(cpu_action==1.)[0][0]
                    #print(action)
                    if isinstance(action_dict[action],
                        ActionSubmit):  # TODO Could be performance improved by using max_key in utils
                        action_dict[action].init_task_name(self.env.forwarded_task_names[i])
                    self.agents[i].send(action)

                dones = []
                # Recieve all action-requests
                for i,agent in enumerate(self.agents):
                    responses[i] = agent.receive()
                    dones.append(responses[i]["type"] != "request-action")
                #next_states, rewards, dones, _ = self.env.step(actions)

                next_states = []
                if all(dones) or step == max_steps - 1:
                    dones = [1 for _ in range(self.num_agents)]
                    rewards = [torch.tensor([[0]]) for _ in range(self.num_agents)]
                    next_states = [None for _ in range(self.num_agents)]
                    #self.replay_buffer.push(states, actions, rewards, next_states, dones)
                    episode_rewards.append(episode_reward)
                    print("episode: {}  |  reward: {}  \n".format(episode, np.round(episode_reward, decimals=4)))
                    break
                else:
                    dones = [0 for _ in range(self.num_agents)]
                    rewards = []
                    for i, agent in enumerate(self.agents):
                        agent.update_env(responses[i])
                        #action = np.where(actions[i].==1.)[0][0]
                        #if responses[i]["content"]["percept"]["lastActionResult"]=="success" and 0<action<5:
                        #    agent.update_coords(action)

                        next_states.append(agent.get_state())
                        last_last_action_and_param = (last_lastAction[i], last_lastAction_param[i])
                        rew = calc_reward_v2(responses[i]['content']['percept'], last_task_names[i], last_tasks[i], attached_cords_in_last_response[i], last_last_action_and_param)
                        attached_cords_in_last_response[i] = get_attached_blocks(responses[i]['content']['percept']['things'],
                                                                              responses[i]['content']['percept']['attached'], cords=True)
                        last_lastAction[i] = responses[i]['content']['percept']['lastAction']
                        last_lastAction_param[i] =  responses[i]['content']['percept']['lastActionParams'][0]
                        last_task_names[i] = self.env.forwarded_task_names[i]
                        last_tasks[i] = self.env.forwarded_task[i]

                        cpu_action = actions[i].cpu().numpy()
                        action = np.where(cpu_action==1.)[0][0]
                        #selected_actions.append(action)
                        selected_action_dict[action].append(rew)
                        rewards.append(torch.tensor([[rew]]))

                        episode_reward += np.mean(rewards)

                    self.replay_buffer.push(states, actions, rewards, next_states, dones)
                    states = next_states

                    if len(self.replay_buffer) > batch_size:
                        self.update(batch_size)

            self.env.kill_server()
            #####
            self.log.save_rewards(episode_rewards)
            self.log.save_actions(selected_action_dict)
            #####
            if episode % 10 == 0:
                self.plot_rewards(episode_rewards, episode)
                self.plot_double_action(selected_action_dict, episode)

    def plot_rewards(self, rewards, name):
        plt.clf()
        plt.plot(rewards)
        plt.title('Training Avg Rewards')
        plt.xlabel('Episode number')
        plt.ylabel('Average Reward')
        plt.savefig(f"plots/Rewards_reward2_{name}.png")

    def plot_double_action(self, actions, name):

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
