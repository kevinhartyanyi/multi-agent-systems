from loq import *
from plots import *

name = "test"
log = Log(name=name)
rewards = log.load_rewards()
actions = log.load_actions()

plot_rewards(rewards=rewards, name=f"{name}_full_rewards_plot")
plot_double_action(actions, f"{name}_full_actions_plot")