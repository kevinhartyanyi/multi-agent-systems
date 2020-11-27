import torch
import numpy as np

from agent import DDPGAgent
from utils import MultiAgentReplayBuffer

from ma_action_classes import action_dict
from utils import calc_reward_v2, get_attached_blocks

class MADDPG:

    def __init__(self, env, buffer_maxlen):
        self.env = env
        self.num_agents = env.n
        self.replay_buffer = MultiAgentReplayBuffer(self.num_agents, buffer_maxlen)
        self.agents = [DDPGAgent(self.env, i) for i in range(self.num_agents)]

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
                _, response = agent.reset()
                responses.append(response)
                agent.update_env(response)
                attached_cords_in_last_response.append([])
                last_lastAction.append(None)
                last_lastAction_param.append(None)
                last_task_names.append([])
                last_tasks.append([])
                states.append(agent.get_state())
            time.sleep(5) # Wait to initialize
            episode_reward = 0
            for step in range(max_steps):
                actions = self.get_actions(states)
                # Send all actions
                for i,action in enumerate(actions):
                    if isinstance(action_dict[action],
                        ActionSubmit):  # TODO Could be performance improved by using max_key in utils
                        action_dict[action].init_task_name(env.forwarded_task_names[i])
                    self.agents[i].send(action)

                dones = []
                # Recieve all action-requests
                for i,agent in enumerate(self.agents):
                    responses[i] = agent.receive()
                    dones.append(response[i]["type"] != "request-action")
                #next_states, rewards, dones, _ = self.env.step(actions)
                episode_reward += np.mean(rewards)

                if all(dones) or step == max_steps - 1:
                    dones = [1 for _ in range(self.num_agents)]
                    rewards = [torch.tensor([[0]]) for _ in range(self.num_agents)]
                    next_states = [None for _ in range(self.num_agents)]
                    self.replay_buffer.push(states, actions, rewards, next_states, dones)
                    episode_rewards.append(episode_reward)
                    print("episode: {}  |  reward: {}  \n".format(episode, np.round(episode_reward, decimals=4)))
                    break
                else:
                    dones = [0 for _ in range(self.num_agents)]
                    rewards = []
                    for i, agent in enumerate(self.agents):
                        last_last_action_and_param[i] = (last_lastAction[i], last_lastAction_param[i])
                        rew = calc_reward_v2(response['content']['percept'], last_task_names[i], last_tasks[i], attached_cords_in_last_response[i], last_last_action_and_param[i])
                        attached_cords_in_last_response[i] = get_attached_blocks(response['content']['percept']['things'],
                                                                              response['content']['percept']['attached'], cords=True)
                        last_lastAction[i] = response['content']['percept']['lastAction']
                        last_lastAction_param[i] =  response['content']['percept']['lastActionParams'][0]
                        last_task_names[i] = env.forwarded_task_names[i]
                        last_tasks[i] = env.forwarded_task[i]

                        collect_rewards.append(rew)
                        selected_actions.append(action.item())
                        selected_action_dict[action.item()].append(rew)
                        rewards.append(torch.tensor([[rew]])

                    self.replay_buffer.push(states, actions, rewards, next_states, dones)
                    states = next_states

                    if len(self.replay_buffer) > batch_size:
                        self.update(batch_size)
