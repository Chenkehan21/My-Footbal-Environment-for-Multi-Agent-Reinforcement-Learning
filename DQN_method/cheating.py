import torch

import os
import sys
sys.path.append(os.path.dirname(__file__) + os.sep + '../')

import random

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
    total_steps = 0

    agent_pos = {}

    for i in range(100):
        print(i)
        all_state = env.reset()
        while True:
            actions, _, AI_actions, _, AI_action, _ = agents.get_actions(all_state, trained_defend_net, trained_attack_net)
            for i in range(1, env.n_agents + 1):
                agent_pos[i] = env.agents[i].pos
            next_state, rewards, done, info = env.step(actions)
            total_steps += 1

            while AI_actions.pop(0) and info['winner'] == 'defend':
                for i in range(1, env.n_agents + 1):
                    env.agents[i].pos = agent_pos[i]
                env.update_map()
                new_AI_action = AI_actions[0]
                for idx, action in enumerate(actions):
                    if action == AI_action:
                        actions[idx] = new_AI_action

                next_state, rewards, done, info = env.step(actions)

            for rew in rewards:
                if rew.team == agents.train_team:
                    total_reward += rew.reward
            
            if done:
                print("game done winner: ", info['winner'])
                if info['winner'] == agents.train_team:
                        total_win_times += 1
                break
            all_state = next_state

    mean_reward = total_reward / 100
    win_rate = total_win_times / 100
    mean_steps = total_steps / 100
    print("mean reward: %.3f" % mean_reward, " | win rate: ", win_rate, " | mean steps: ", mean_steps)


if __name__ == "__main__":
    cheat()