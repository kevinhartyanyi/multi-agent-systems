from random_agent import *
from reinforce_agent import *
from server_env_new import *


env = Server()

class AgentCommunication:
    def __init__(self):
        self.agents = []

    def init_agents(self, env, agent_names, EXTRA_SMART = False):
        agent_id = 1

        self.agents = []

        for i in range(len(agent_names)):
            if EXTRA_SMART:
                agent = Reinforce_Agent(agent_names[i], agent_id, env)
            else:
                agent = Random_Agent(agent_names[i], agent_id, env)

            agent.connect()
            assert agent.init_agent()  # auth-response

            self.agents.append(agent)

        #response = agent.receive()  # sim-start (vision, step)
        #response = agent.receive()  # request-action
        #print(response)
    
    def send(self, actions):
        for i, agent in enumerate(self.agents):
            print("Comm:", agent.name)
            response = agent.receive()  # sim-start (vision, step)
            #response = agent.receive()  # request-action
            #agent.send(actions[i])
            #response = agent.receive()

comm = AgentCommunication()

comm.init_agents(env, [f"agentA{num + 1}" for num in range(3)])
comm.send([1,2,3])