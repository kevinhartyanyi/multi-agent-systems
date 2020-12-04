from maddpg import MADDPG
from multiagentenv import MultiAgentEnv
import ma_assumptions
#https://github.com/cyoon1729/Multi-agent-reinforcement-learning/tree/master/MADDPG

if __name__ == "__main__":
    num_agents = 3
    num_episodes = 11
    num_steps = 10

    env = MultiAgentEnv(num_agents)
    ma_controller = MADDPG(env, 1000000)
    ma_controller.run(num_episodes, num_steps,32, monitor=True)
