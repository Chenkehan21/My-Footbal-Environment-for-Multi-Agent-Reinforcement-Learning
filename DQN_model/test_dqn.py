import os
import sys
import random
from numpy.lib.function_base import copy
sys.path.append(os.path.dirname(__file__) + os.sep + '../')

import torch
from football_env import Football_Env
import numpy as np
from utils import *
from DQN_model import DQN

ATTACK_PATH = '/home/chenkehan/RESEARCH/codes/experiment4/DQN_model/res_attack5/football_0.710_59.780.dat'
DEFEND_PATH = '/home/chenkehan/RESEARCH/codes/experiment4/DQN_model/res_defend4/football_0.860_98.000.dat'


def check_action(env, action, team):
    for agent in env.agents.values():
        if team == "attack" and agent.team == team:
            actions = list(range(7))
            if agent.pos[0] == 0:
                actions.remove(1)
            if agent.pos[0] == env.court_height - 1:
                actions.remove(2)
            if agent.pos[1] == 0:
                actions.remove(3)
            if agent.pos[1] == env.court_width - 1:
                actions.remove(4)

            shoot_pos = agent.can_shoot()
            if shoot_pos:
                shoot_success_rate = agent.success_rate[shoot_pos[0]]
                if (agent.posses_ball and shoot_pos and random.random() < shoot_success_rate) != True:
                    actions.remove(6)
            else:
                actions.remove(6)
            
            if action in actions:
                return True

        if team == "defend" and agent.team == team:
            actions = list(range(5))
            if agent.pos[0] == 0:
                actions.remove(1)
            if agent.pos[0] == env.court_height - 1:
                actions.remove(2)
            if agent.pos[1] == 0:
                actions.remove(3)
            if agent.pos[1] == env.court_width - 1:
                actions.remove(4)
            
            if action in actions:
                return True
    return False


def test(env, train_team, device, 
         use_trained_defend_net, use_trained_attack_net,
         defender_net_path, attacker_net_path):
    print("start testing")

    if train_team == "attack":
        AI_team = "defend"
    else:
        AI_team = "attack"

    total_win_times = 0
    total_reward = 0.0
    total_steps = 0

    all_state = env.reset()
    if train_team:
        state = handle_obs(all_state, train_team)
        input_shape = len(state)

    if train_team == "attack":
        output_shape = env.attack_action_space_n
    if train_team == "defend":
        output_shape = env.defend_action_space_n
        

    if use_trained_defend_net:
        output_shape = env.defend_action_space_n
        trained_defend_net = DQN(input_shape, output_shape).to(device)
        trained_defend_net.eval()
        if defender_net_path:
            trained_defend_net.load_state_dict(torch.load(defender_net_path))

    if use_trained_attack_net:
        output_shape = env.attack_action_space_n
        trained_attack_net = DQN(input_shape, output_shape).to(device)
        trained_attack_net.eval()
        if attacker_net_path:
            trained_attack_net.load_state_dict(torch.load(attacker_net_path))

    total_reward = 0.0
    # all_state = self.env.reset()

    for i in range(100):
        print(i)
        all_state = env.reset()
        while not env.Done:
            if train_team:
                state = handle_obs(all_state, train_team)
                state_a = np.array(state, copy=False) # add an dimension for BATCH_SIZE!
                state_v = torch.tensor(state_a, dtype=torch.float).to(device)
            
            AI_state = []
            if train_team == 'attack':
                AI_state = handle_obs(all_state, 'defend')
            if train_team == 'defend':
                AI_state = handle_obs(all_state, 'attack')


            if len(AI_state) > 0:
                AI_state_a = np.array(AI_state, copy=False)
                AI_state_v = torch.tensor(AI_state_a, dtype=torch.float).to(device)
                    
            actions = []
            trainer_action, AI_action = None, None
            all_actions = env.sample_actions()
            if train_team == 'attack':
                if use_trained_attack_net:
                    trainer_q_values = trained_attack_net(state_v)
                    sorted_q_values, index = trainer_q_values.sort(descending=True)
                    trainer_actions = index.tolist()
                    while check_action(env, trainer_actions[0], train_team) == False:
                        trainer_actions.pop(0)
                    trainer_action = trainer_actions[0]

                    # _, action_v = torch.max(trainer_q_values, dim=0)
                    # trainer_action = int(action_v.item())
                else:
                    for action_info in all_actions:
                        if action_info.team == 'attack':
                            action = action_info.action
                            action_team = action_info.team
                            while check_action(env, action, action_team) == False:
                                action = random.choice(list(range(7)))
                            trainer_action = action

                if use_trained_defend_net:
                    AI_q_values = trained_defend_net(AI_state_v)
                    # _, AI_action_v = torch.max(AI_q_values, dim=0)
                    # AI_action = int(AI_action_v.item())
                    sorted_q_values, index = AI_q_values.sort(descending=True)
                    AI_actions = index.tolist()
                    while check_action(env, AI_actions[0], AI_team) == False:
                        # print("AI actions: ", AI_actions)
                        AI_actions.pop(0)
                    AI_action = AI_actions[0]
                else:
                    for action_info in all_actions:
                        if action_info.team == 'defend':
                            action = action_info.action
                            action_team = action_info.team
                            while check_action(env, action, action_team) == False:
                                action = random.choice(list(range(7)))
                            AI_action = action
                
                for action in all_actions:
                    if action.team == train_team:
                        actions.append(trainer_action)
                    else:
                        actions.append(AI_action)
            
            if train_team == 'defend':
                if use_trained_defend_net:
                    trainer_q_values = trained_defend_net(state_v)
                    sorted_q_values, index = trainer_q_values.sort(descending=True)
                    trainer_actions = index.tolist()
                    while check_action(env, trainer_actions[0], train_team) == False:
                        trainer_actions.pop(0)
                    trainer_action = trainer_actions[0]
                    # _, action_v = torch.max(trainer_q_values, dim=0)
                    # trainer_action = int(action_v.item())
                else:
                    for action_info in all_actions:
                        if action_info.team == 'defend':
                            action = action_info.action
                            action_team = action_info.team
                            while check_action(env, action, action_team) == False:
                                action = random.choice(list(range(7)))
                            trainer_action = action

                if use_trained_attack_net:
                    AI_q_values = trained_attack_net(AI_state_v)
                    sorted_q_values, index = AI_q_values.sort(descending=True)
                    AI_actions = index.tolist()
                    while check_action(env, AI_actions[0], AI_team) == False:
                        # print("AI actions: ", AI_actions)
                        AI_actions.pop(0)
                    AI_action = AI_actions[0]
                    # _, AI_action_v = torch.max(AI_q_values, dim=0)
                    # AI_action = int(AI_action_v.item())
                else:
                    for action_info in all_actions:
                        if action_info.team == 'attack':
                            action = action_info.action
                            action_team = action_info.team
                            while check_action(env, action, action_team) == False:
                                action = random.choice(list(range(7)))
                            AI_action = action

                for action in all_actions:
                    if action.team == train_team:
                        actions.append(trainer_action)
                    else:
                        actions.append(AI_action)
            
            if len(actions) == 0:
                for action_info in all_actions:
                    action = action_info.action
                    action_team = action_info.team
                    while check_action(env, action, action_team) == False:
                        action = random.choice(list(range(7)))
                    actions.append(action)
            
            # print("actions: ", actions)
            next_state, rewards, done, info = env.step(actions)
            total_steps += 1

            for rew in rewards:
                if rew.team == train_team:
                    total_reward += rew.reward
            
            if done:
                print("game done winner: ", info['winner'])
                if info['winner'] == train_team:
                        total_win_times += 1
                break

            all_state = next_state

    mean_reward = total_reward / 100
    win_rate = total_win_times / 100
    mean_steps = total_steps / 100
    print("mean reward: %.3f" % mean_reward, " | win rate: ", win_rate, " | mean steps: ", mean_steps)


if __name__ == "__main__":
    env = Football_Env(agents_left=[1], agents_right=[2],
    max_episode_steps=500, move_reward_weight=1.0,
    court_width=23, court_height=20, gate_width=6)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test(env, 'attack', device=device,
         use_trained_attack_net=True, use_trained_defend_net=True,
        defender_net_path=DEFEND_PATH, attacker_net_path=ATTACK_PATH)