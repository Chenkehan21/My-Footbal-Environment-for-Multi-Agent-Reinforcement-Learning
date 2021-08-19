from math import inf
import os
import sys
import random
import time
from numpy.lib.function_base import copy
sys.path.append(os.path.dirname(__file__) + os.sep + '../')

import torch
from football_env import Football_Env
import numpy as np
from utils import *
from agents import Agents

ATTACK_PATH1 = '/home/chenkehan/RESEARCH/codes/experiment4/DQN_method/res_attack5/football_0.710_59.780.dat'
ATTACK_PATH2 = '/home/chenkehan/RESEARCH/codes/experiment4/DQN_method/res_attack5/football_0.890_72.300.dat'
# ATT.ACK_PATH = '/home/chenkehan/RESEARCH/codes/experiment4/DQN_method/res_attack6/football_93.650_0.890.dat'
DEFEND_PATH1 = '/home/chenkehan/RESEARCH/codes/experiment4/DQN_method/res_defend4/football_0.860_98.000.dat'
DEFEND_PATH2 = '/home/chenkehan/RESEARCH/codes/experiment4/DQN_method/res_defend4/football_0.830_100.000.dat'
# DEFEND_PATH = '/home/chenkehan/RESEARCH/codes/experiment4/DQN_method/res_defend4/football_0.700_100.000.dat'

def test():
    env = Football_Env(agents_left=[1, 2], agents_right=[3, 4],
    max_episode_steps=500, move_reward_weight=1.0,
    court_width=23, court_height=20, gate_width=6)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agents = Agents(env, 'attack', device=device,
         use_trained_attack_net=True, use_trained_defend_net=True,
        defender_net_path=[DEFEND_PATH1, DEFEND_PATH2], attacker_net_path=[ATTACK_PATH1, ATTACK_PATH2])
    
    trained_defend_net, trained_attack_net = agents.get_nets()

    total_reward = 0.0
    total_win_times = 0
    total_tie_times = 0
    total_defend_times = 0
    total_steps = 0

    for i in range(100):
        print(i)
        all_state = env.reset()
        while True:
            actions, _, _, _, _, _, _ = agents.get_actions(all_state, trained_defend_net, trained_attack_net)
            next_state, rewards, done, info = env.step(actions)
            total_steps += 1

            for rew in rewards:
                if rew.team == agents.train_team:
                    total_reward += rew.reward
            
            if done:
                print("game done winner: ", info['winner'], "\n")
                if info['winner'] == agents.train_team:
                    total_win_times += 1
                if info['winner'] == 'tie':
                    total_tie_times += 1
                if info['winner'] == 'defend':
                    total_defend_times += 1
                break
            all_state = next_state

    mean_reward = total_reward / 100
    win_rate = total_win_times / 100
    tie_rate = total_tie_times / 100
    defend_rate = total_defend_times / 100
    mean_steps = total_steps / 100
    print("mean reward: %.3f" % mean_reward, " | win rate: ", win_rate, " | tie rate", tie_rate, " | defend rate", defend_rate, " | mean steps: ", mean_steps)


if __name__ == "__main__":
    test()