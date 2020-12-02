import pandas as pd
import os
from natsort import natsorted
import glob
import pickle

class Log:
    def __init__(self, name: str):
        self.name = name
        self.reward_col = "rewards"
        self.actions = "actions"
        self.dir = "train_logs"
        self.__check_files()
        self.saveFileNameRewards = os.path.join(self.dir, f"{name}_{self.reward_col}_{self.num}")
        self.saveFileNameActions = os.path.join(self.dir, f"{name}_{self.actions}_{self.num}")

    def __check_files(self):
        fn = glob.glob(os.path.join(self.dir, self.name + "*"))
        files = natsorted(fn)
        if len(files) > 0:
            self.num = int(files[-1].split("_")[-1]) + 1
        else:
            self.num = 0
        print("Log num:", self.num)

    def save_rewards(self, rewards):
        df = pd.DataFrame(data={self.reward_col :rewards, })
        df.to_csv(self.saveFileNameRewards, sep=',', index=False)

    def save_actions(self, actions):
        with open(self.saveFileNameActions + '.pkl', 'wb') as f:
            pickle.dump(actions, f, pickle.HIGHEST_PROTOCOL)

    def load_actions(self):
        fn = glob.glob(os.path.join(self.dir, self.name + f"_{self.actions}*"))
        files = natsorted(fn)
        re = {}  # Create an empty dictionary
        for f in files:
            with open(f, 'rb') as p:
                tmp_dict = pickle.load(p)
            for k, v in tmp_dict.items():
                if k not in re:
                    re[k] = v
                else:
                    for e in v:
                        re[k].append(e)
        return re

    def load_rewards(self):
        fn = glob.glob(os.path.join(self.dir, self.name + f"_{self.reward_col}*"))
        files = natsorted(fn)
        li = []
        for f in files:
            tmp_df = pd.read_csv(f, index_col=None, header=0)
            li.append(tmp_df)
        df = pd.concat(li, axis=0, ignore_index=True)
        return df