from hashlib import new
import torch

import os
import sys
sys.path.append(os.path.dirname(__file__) + os.sep + '../')

import random
import time

from football_env import Football_Env
from utils import *
from agents import Agents


ATTACK_PATH1 = '/home/chenkehan/RESEARCH/codes/experiment4/DQN_method/res_attack5/football_0.710_59.780.dat'
ATTACK_PATH2 = '/home/chenkehan/RESEARCH/codes/experiment4/DQN_method/res_attack5/football_0.890_72.300.dat'

DEFEND_PATH1 = '/home/chenkehan/RESEARCH/codes/experiment4/DQN_method/res_defend4/football_0.860_98.000.dat'
DEFEND_PATH2 = '/home/chenkehan/RESEARCH/codes/experiment4/DQN_method/res_defend4/football_0.830_100.000.dat'

TEST_NUM = 1000

def cheat():
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

    agent_pos = {}

    for i in range(TEST_NUM):
        print(i)
        # time.sleep(2)
        all_state = env.reset()
        while True:
            actions, trainer_actions, AI_actions, trainer_action, AI_action, _, _ = agents.get_actions(all_state, trained_defend_net, trained_attack_net)
            for i in range(1, env.n_agents + 1):
                agent_pos[i] = env.agents[i].pos

            next_state, rewards, done, info = env.step(actions)
            total_steps += 1
            while done and info['winner'] == 'defend' and trainer_actions[info['lose_id'] - 1].pop(0) >= 0 and trainer_actions[info['lose_id'] - 1]:
                for i in range(1, env.n_agents + 1):
                    env.agents[i].pos = agent_pos[i]
                env.update_map()
                new_trainer_action = trainer_actions[info['lose_id'] - 1][0]
                for idx, action in enumerate(actions):
                    if action == trainer_action[info['lose_id'] - 1]:
                        actions[idx] = new_trainer_action
                next_state, rewards, done, info = env.step(actions)
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

    mean_reward = total_reward / TEST_NUM
    win_rate = total_win_times / TEST_NUM
    tie_rate = total_tie_times / TEST_NUM
    defend_rate = total_defend_times / TEST_NUM
    mean_steps = total_steps / TEST_NUM
    print("mean reward: %.3f" % mean_reward, " |win rate: ", win_rate, " |tie rate: ", tie_rate, " |defend rate: ", defend_rate, " |mean steps: ", mean_steps)


if __name__ == "__main__":
    cheat()