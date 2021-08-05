import os
import sys
sys.path.append(os.path.dirname(__file__) + os.sep + '../')

import torch
from football_env import Football_Env
import numpy as np
from utils import *
from DQN_model import DQN

PATH = '/home/chenkehan/RESEARCH/codes/experiment4/DQN_model/res_win_rate/football_0.880_66.500.dat'


def test(env, train_team, device):
    print("start testing")
    state = env.reset()
    state = handle_obs(state, train_team)

    if train_team == "attack":
        action_space = env.attack_action_space_n
    if train_team == "defend":
        action_space = env.defend_action_space_n
    
    input_shape = len(state)
    output_shape = action_space 
    net = DQN(input_shape, output_shape).to(device)
    net.eval()
    net.load_state_dict(torch.load(PATH))

    # state = torch.tensor([13.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  7.,  0.,  8.,
    #      0.,  9.,  0., 10.,  0., 11.,  0., 12.,  0., 13.,  0.,  7., 22.,  8.,
    #     22.,  9., 22., 10., 22., 11., 22., 12., 22., 13., 22.,  1.], dtype=torch.float, device='cuda')
    # q_value = net(state)
    # print(q_value)

    total_win_times = 0
    total_reward = 0.0
    for i in range(100):
        state = env.reset()
        while True:
            state = handle_obs(state, train_team)
            state_a = np.array(state, copy=False) # add an dimension for BATCH_SIZE!
            state_v = torch.tensor(state_a, dtype=torch.float).to(device)
            # print(state_v)
            q_values = net(state_v) # since we put in state_v, batch_size=1 so q_values's size is [1, action_space.n]
            # print(q_values)
            _, action_v = torch.max(q_values, dim=0)
            trainer_action = int(action_v.item())
            # print("trainer action: ", trainer_action)
            total_actions = env.sample_actions()
            actions = []
            for action in total_actions:
                if action.team == train_team:
                    actions.append(trainer_action)
                else:
                    actions.append(action.action)
            # print("actions: ", actions)
            # step action
            next_state, rewards, done, info = env.step(actions)
            for rew in rewards:
                if rew.team == train_team:
                    total_reward += rew.reward
            
            if done:
                print("winner: ", info['winner'])
                if info['winner'] == 'attack':
                    total_win_times += 1
                break

            state = next_state
    
    mean_reward = total_reward / 100
    win_rate = total_win_times / 100
    print("mean reward: %.3f" % mean_reward, " | win rate: ", win_rate)

if __name__ == "__main__":
    env = Football_Env(agents_left=[1], agents_right=[2],
    max_episode_steps=50000, move_reward_weight=1.0,
    court_width=23, court_height=20, gate_width=6)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test(env, 'defend', device)