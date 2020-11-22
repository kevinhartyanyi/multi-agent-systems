from random_agent import *
from reinforce_agent import *
from server_env_new import *
import torch

import time
from queue import Queue


class AgentCommunication:
    def __init__(self):
        self.agents = []
        self.responses = []
        self.server_comm = Queue(0)

    def init_agents(self, env, agent_names, EXTRA_SMART = False):
        """
        Currently doesn't work with Random_Agent, because it has different state shape
        Could be improved with threading
        """
        def register_agent(agent):
            agent.connect()
            assert agent.init_agent()  # auth-response  
            agent.receive() # SIM-START

        agent_id = 1
        self.agents = []

        # REGISTER
        for i in range(len(agent_names)):
            if EXTRA_SMART:
                agent = Reinforce_Agent(agent_names[i], agent_id, env)
            else:
                agent = Random_Agent(agent_names[i], agent_id, env)

            register_agent(agent)   
            self.agents.append(agent)

    def init_agents_subprocess(self, env, agent_names, process, EXTRA_SMART = False):
        """
        Currently doesn't work with Random_Agent, because it has different state shape
        Could be improved with threading
        """
        def register_agent(agent):
            #time.sleep(5)
            agent.connect()
            assert agent.init_agent()  # auth-response
            #time.sleep(2)
            process.stdin.write(b'\n')
            process.stdin.flush()
            agent.receive() # SIM-START

        agent_id = 1
        self.agents = []

        # REGISTER
        for i in range(len(agent_names)):
            print("Init agent:",agent_names[i])
            if EXTRA_SMART:
                agent = Reinforce_Agent(agent_names[i], agent_id, env)
            else:
                agent = Random_Agent(agent_names[i], agent_id, env)

            register_agent(agent)   
            self.agents.append(agent)
    
    def update_env(self):
        states = []
        for i, agent in enumerate(self.agents):
            agent.update_env(self.responses[i])
            next_state = torch.from_numpy(agent.get_state()).type(torch.DoubleTensor).unsqueeze(0).unsqueeze(0)
            states.append(next_state)
        return states

    

    def receive(self):
        self.responses = []
        for agent in self.agents:
            self.responses.append(agent.receive()) # request-action

    def agents_comm(self, actions):
        self.responses = []
        done = False
        print("Comm start")
        for i, agent in enumerate(self.agents):
            agent.send(actions[i])
            response = agent.receive() # request-action
            print(response['content']['percept']['lastActionParams'])
            self.responses.append(response) 

            if response["type"] != "request-action":
                done = True
        print("Comm end")
        return done
    
    def reset_agents(self):
        for agent in self.agents:
            agent.reset()


comm = AgentCommunication()
env = Server()
comm.init_agents(env, [f"agentA{num + 1}" for num in range(3)])
comm.agents_comm([1,2,3])
comm.agents_comm([1,2,3])
comm.agents_comm([1,2,3])
