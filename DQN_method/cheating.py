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


ATTACK_PATH = '/home/chenkehan/RESEARCH/codes/experiment4/DQN_method/res_attack5/football_0.710_59.780.dat'
DEFEND_PATH = '/home/chenkehan/RESEARCH/codes/experiment4/DQN_method/res_defend4/football_0.860_98.000.dat'


def cheat():
    env = Football_Env(agents_left=[1], agents_right=[2],
    max_episode_steps=500, move_reward_weight=1.0,
    court_width=23, court_height=20, gate_width=6)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agents = Agents(env, 'attack', device=device,
         use_trained_attack_net=True, use_trained_defend_net=True,
        defender_net_path=DEFEND_PATH, attacker_net_path=ATTACK_PATH)
    
    trained_defend_net, trained_attack_net = agents.get_nets()

    total_reward = 0.0
    total_win_times = 0
    total_tie_times = 0
    total_defend_times = 0
    total_steps = 0

    agent_pos = {}

    for i in range(100):
        print(i)
        # time.sleep(2)
        all_state = env.reset()
        while True:
            actions, trainer_actions, AI_actions, trainer_action, AI_action, _, _ = agents.get_actions(all_state, trained_defend_net, trained_attack_net)
            for i in range(1, env.n_agents + 1):
                agent_pos[i] = env.agents[i].pos
            # print("true actions: ", actions)
            next_state, rewards, done, info = env.step(actions)
            total_steps += 1

            while done and info['winner'] == 'defend' and trainer_actions.pop(0) and trainer_actions:
                # print("+++++++ in loop ++++++++")
                # print("agent pos: ", agent_pos)
                for i in range(1, env.n_agents + 1):
                    env.agents[i].pos = agent_pos[i]
                env.update_map()
                new_trainer_action = trainer_actions[0]
                # print("new_trainer_action: ", new_trainer_action)
                for idx, action in enumerate(actions):
                    if action == trainer_action:
                        actions[idx] = new_trainer_action
                # print("virtual actions: ", actions)
                next_state, rewards, done, info = env.step(actions)
                # for i in range(1, env.n_agents + 1):
                    # print("agent_%d pos: " % env.agents[i].id, env.agents[i].pos, "agent team: ", env.agents[i].team)
            # print("+++++++ out loop ++++++++")

            for rew in rewards:
                if rew.team == agents.train_team:
                    total_reward += rew.reward
            
            if done:
                print("game done winner: ", info['winner'])
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
    print("mean reward: %.3f" % mean_reward, " |win rate: ", win_rate, " |tie rate: ", tie_rate, " |defend rate: ", defend_rate, " |mean steps: ", mean_steps)


if __name__ == "__main__":
    cheat()