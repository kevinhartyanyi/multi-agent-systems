import matplotlib.pyplot as plt
from ma_action_classes import action_dict
import numpy as np

def plot_rewards(rewards, name):
    plt.clf()
    plt.plot(rewards)
    plt.title('Training Avg Rewards')
    plt.xlabel('Episode number')
    plt.ylabel('Average Reward')
    plt.savefig(f"plots/Rewards_{name}.png")

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