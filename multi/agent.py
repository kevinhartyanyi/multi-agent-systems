import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import numpy as np

from model import CentralizedCritic, Actor

import json, socket

import ma_assumptions
from ma_message_classes import AuthRequest


class DDPGAgent:

    def __init__(self, env, agent_id, agent_pass=10, actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=1e-2):
        self.env = env
        self.agent_id = agent_id
        self.agent_pass = agent_pass
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.tau = tau

        self.request_id = 0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.map = np.array([[0, 0, 0, 0, 0]])

        self.state = None

        self.step = np.array([0, ma_assumptions.STEP_NUM]) # current step, assumptions.STEP_NUM (Will certainly be updated, no need to set to assumptions.IGNORE)
        self.dispensers = ma_assumptions.IGNORE * np.ones((ma_assumptions.DISPENSER_NUM, 3))
        self.walls = ma_assumptions.IGNORE * np.ones((ma_assumptions.WALL_NUM, 2))

        num_inputs = self.step.size + self.dispensers.size + self.walls.size
        print("NUM_INPUTS:", num_inputs)
        num_actions = len(action_dict)

        self.device = "cpu"
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.device = "cuda"

        self.obs_dim = num_inputs
        self.action_dim = num_actions
        self.num_agents = self.env.n

        self.critic_input_dim = int(np.sum([env.observation_space(agent)+num_inputs for agent in range(env.n)]))
        self.actor_input_dim = self.obs_dim

        self.critic = CentralizedCritic(self.critic_input_dim, self.action_dim * self.num_agents).to(self.device)
        self.critic_target = CentralizedCritic(self.critic_input_dim, self.action_dim * self.num_agents).to(self.device)
        self.actor = Actor(self.actor_input_dim, self.action_dim).to(self.device)
        self.actor_target = Actor(self.actor_input_dim, self.action_dim).to(self.device)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        self.MSELoss = nn.MSELoss()
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

    def get_action(self, state):
        state = np.hstack([x.flatten() for x in self.state] )
        state = autograd.Variable(torch.from_numpy(state).float().squeeze(0)).to(self.device)
        action = self.actor.forward(state)
        action = self.onehot_from_logits(action)

        return action

    def onehot_from_logits(self, logits, eps=0.0):
        # get best (according to current policy) actions in one-hot form
        argmax_acs = (logits == logits.max(0, keepdim=True)[0]).float()
        if eps == 0.0:
            return argmax_acs
        # get random actions in one-hot form
        rand_acs = Variable(torch.eye(logits.shape[1])[[np.random.choice(
            range(logits.shape[1]), size=logits.shape[0])]], requires_grad=False)
        # chooses between best and random actions using epsilon greedy
        return torch.stack([argmax_acs[i] if r > eps else rand_acs[i] for i, r in
                            enumerate(torch.rand(logits.shape[0]))])

    def update(self, indiv_reward_batch, indiv_obs_batch, global_state_batch, global_actions_batch, global_next_state_batch, next_global_actions):
        """
        indiv_reward_batch      : only rewards of agent i
        indiv_obs_batch         : only observations of agent i
        global_state_batch      : observations of all agents are concatenated
        global actions_batch    : actions of all agents are concatenated
        global_next_state_batch : observations of all agents are concatenated
        next_global_actions     : actions of all agents are concatenated
        """
        indiv_reward_batch = torch.FloatTensor(indiv_reward_batch).to(self.device)
        indiv_reward_batch = indiv_reward_batch.view(indiv_reward_batch.size(0), 1).to(self.device)
        indiv_obs_batch = torch.FloatTensor(indiv_obs_batch).to(self.device)
        global_state_batch = torch.FloatTensor(global_state_batch).to(self.device)
        global_actions_batch = torch.stack(global_actions_batch).to(self.device)
        global_next_state_batch = torch.FloatTensor(global_next_state_batch).to(self.device)
        next_global_actions = next_global_actions

        # update critic
        self.critic_optimizer.zero_grad()

        curr_Q = self.critic.forward(global_state_batch, global_actions_batch)
        next_Q = self.critic_target.forward(global_next_state_batch, next_global_actions)
        estimated_Q = indiv_reward_batch + self.gamma * next_Q

        critic_loss = self.MSELoss(curr_Q, estimated_Q.detach())
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        # update actor
        self.actor_optimizer.zero_grad()

        policy_loss = -self.critic.forward(global_state_batch, global_actions_batch).mean()
        curr_pol_out = self.actor.forward(indiv_obs_batch)
        policy_loss += -(curr_pol_out**2).mean() * 1e-3
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.actor_optimizer.step()

    def target_update(self):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    def update_env(self, msg):
        self.state = self.env.update(msg['content']['percept'])
        observation_vector = self.state[0]
        #print("Observation vector: ", observation_vector)
        for obs in observation_vector:
            ind = find_coord_index(self.map, obs[:2])
            # print("Check: ", obs[:2])
            # print("Index", ind)
            if ind == -1:  # New Entry
                self.map = np.append(self.map, np.array([obs]), axis=0)
            else:  # Update
                self.map[ind] = obs

        # Update walls
        new_walls = self.map[(self.map[:,2] == 0) & (self.map[:,3] == 0) & (self.map[:,4] == 2)][:,:2]

        empty_wall = np.where(self.walls[:,0] == assumptions.IGNORE)[0]
        new_walls_count = 0
        while new_walls_count < len(new_walls) and 0 < len(empty_wall):
            if new_walls[new_walls_count].tolist() not in self.walls.tolist():
                self.walls[empty_wall[0]] = new_walls[new_walls_count]
                empty_wall = empty_wall[1:]
            new_walls_count += 1

        # Update dispensers
        new_dispensers = self.map[(self.map[:,2] == 3)][:,:4]
        empty_dispensers = np.where(self.dispensers[:,0] == assumptions.IGNORE)[0]
        new_dispensers_count = 0
        while new_dispensers_count < len(new_dispensers) and 0 < len(empty_dispensers):
            if new_dispensers[new_dispensers_count][:2].tolist() not in self.dispensers[:,:2].tolist():
                self.dispensers[empty_dispensers[0]][:2] = new_dispensers[new_dispensers_count][:2]
                self.dispensers[empty_dispensers[0]][2] = new_dispensers[new_dispensers_count][3]
                empty_dispensers = empty_dispensers[1:]
            new_dispensers_count += 1

        # Final state of state ;)
        self.state = np.array([data for data in self.state] + [self.step, self.walls, self.dispensers])
        #print("State shape:", self.state.shape)

    def connect(self, host: str = "127.0.0.1", port: int = 12300):

        connected = False
        wait_sec = 2
        while not connected:
            try:
                self.sock.connect((host, port))
            except ConnectionRefusedError:
                print(f"Connection Refused. Trying again after {wait_sec} seconds")
                time.sleep(0.1)
            else:
                connected = True

    def init_agent(self):
        agent_message = AuthRequest("agentA"+str(self.agent_id), self.agent_pass)
        msg = agent_message.msg()
        #print(f"Sending: {msg}")
        self.sock.sendall(msg.encode())
        response = json.loads(self.sock.recv(4096).decode("ascii").rstrip('\x00'))
        #print(f"Response: {response}")
        return True if response["content"]["result"] == "ok" else False

    def reset(self):
        self.request_id = 0

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.map = np.array([[0, 0, 0, 0, 0]])

        # Network parameters
        self.state = None
        self.step = np.array([0, ma_assumptions.STEP_NUM])
        self.dispensers = ma_assumptions.IGNORE * np.ones((ma_assumptions.DISPENSER_NUM, 3))
        self.walls = ma_assumptions.IGNORE * np.ones((ma_assumptions.WALL_NUM, 2))

        self.connect()
        init = self.init_agent()
        response = self.receive()
        return init, response

    def send(self, action: int):
        """ Send an ActionReply to the server
        """
        agent_message = ActionReply(self.request_id, action_dict[action])
        msg = agent_message.msg()
        #print(f"Sending: {msg}")
        self.sock.sendall(msg.encode())

    def receive(self):
        """ Receive message from the server
        """
        while True:
            recv = self.sock.recv(4096).decode("ascii").rstrip('\x00')
            if recv != "":
                break
        response = json.loads(recv.rstrip('\x00'))
        #print(f"Response: {response}")
        if response['type'] == "request-action":
            self.request_id = response['content']['id']
            self.step[0] = response['content']['step'] # Current steps
            #self.update_env(response)
        ##TODO Check request_action
        return response
