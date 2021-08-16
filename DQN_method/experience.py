import os
import sys
sys.path.append(os.path.dirname(__file__) + os.sep + '../')


import random
import pickle

from football_env import Football_Env
from utils import *
from DQN_method.DQN_model import DQN
from agents import Agents

import torch
from torch.utils.data import Dataset


ATTACK_PATH = '/home/chenkehan/RESEARCH/codes/experiment4/DQN_method/res_attack5/football_0.710_59.780.dat'
DEFEND_PATH = '/home/chenkehan/RESEARCH/codes/experiment4/DQN_method/res_defend4/football_0.860_98.000.dat'

EXPERIENCE_SIZE = 10000


class Experience(Agents):
    def __init__(self, env, train_team, device, 
         use_trained_defend_net, use_trained_attack_net,
         defender_net_path, attacker_net_path, experience_size):
        super(Experience, self).__init__(
            env, train_team, device, 
            use_trained_defend_net, use_trained_attack_net,
            defender_net_path, attacker_net_path)

        self.experience_size = experience_size
        self.trained_defend_net, self.trained_attack_net = self.get_nets()

    def generate_experience(self):
        experience = []
        n, m, k = 0, 0, 0
        while n <= 3000 or m <= 3000 or k <= 2000:
            all_state = self.env.reset()
            while True:
                actions, _, _, _, AI_action, state_v, map_state = self.get_actions(all_state, self.trained_defend_net, self.trained_attack_net)
                if self.train_team == 'attack':
                    experience_state = state_v.to(self.device)
                    # experience_state = torch.FloatTensor(map_state)
                    # experience_state = experience_state.reshape(1, -1).squeeze()
                    experience_action = torch.FloatTensor([AI_action]).to(self.device)
                    if AI_action == 1:
                        n += 1
                        if n <= 3000:
                            experience.append((experience_state, experience_action))
                    elif AI_action == 2:
                        m += 1
                        if m <= 3000:
                            experience.append((experience_state, experience_action))
                    elif AI_action == 4:
                        k += 1
                        experience.append((experience_state, experience_action))
                else:
                    print("train team is defend!")
                    exit()

                next_state, rewards, done, info = self.env.step(actions)

                if done:
                    break

                all_state = next_state
    
        return experience


class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index: int):
        states = self.dataset[index][0]
        actions = self.dataset[index][1]
        return states, actions
    
    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    env = Football_Env(agents_left=[1], agents_right=[2],
    max_episode_steps=500, move_reward_weight=1.0,
    court_width=23, court_height=20, gate_width=6)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    my_experience = Experience(env, 'attack', device=device,
         use_trained_attack_net=True, use_trained_defend_net=True,
        defender_net_path=DEFEND_PATH, attacker_net_path=ATTACK_PATH, experience_size=EXPERIENCE_SIZE)

    experience = my_experience.generate_experience()
    with open('./experience.pkl', 'wb') as f:
        pickle.dump(experience, f)