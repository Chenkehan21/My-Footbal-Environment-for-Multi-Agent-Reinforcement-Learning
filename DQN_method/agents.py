import os
import sys
import random
from numpy.lib.function_base import copy
sys.path.append(os.path.dirname(__file__) + os.sep + '../')

import torch
from football_env import Football_Env
import numpy as np
from utils import *
from DQN_method.DQN_model import DQN

ATTACK_PATH = '/home/chenkehan/RESEARCH/codes/experiment4/DQN_method/res_attack5/football_0.710_59.780.dat'
DEFEND_PATH = '/home/chenkehan/RESEARCH/codes/experiment4/DQN_method/res_defend4/football_0.860_98.000.dat'

class Agents:
    def __init__(self, env, train_team, device, 
         use_trained_defend_net, use_trained_attack_net,
         defender_net_path, attacker_net_path):
        self.env = env
        self.train_team = train_team
        if self.train_team == 'attack':
            self.AI_team = 'defend'
        else:
            self.AI_team = 'attack'
        self.device = device

        self.use_trained_defend_net = use_trained_defend_net
        self.use_trained_attack_net = use_trained_attack_net
        self.defender_net_path = defender_net_path
        self.attacker_net_path = attacker_net_path

    
    def check_action(self, action, team):
        for agent in self.env.agents.values():
            if team == "attack" and agent.team == team:
                actions = list(range(7))
                if agent.pos[0] == 0:
                    actions.remove(1)
                if agent.pos[0] == self.env.court_height - 1:
                    actions.remove(2)
                if agent.pos[1] == 0:
                    actions.remove(3)
                if agent.pos[1] == self.env.court_width - 1:
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
                if agent.pos[0] == self.env.court_height - 1:
                    actions.remove(2)
                if agent.pos[1] == 0:
                    actions.remove(3)
                if agent.pos[1] == self.env.court_width - 1:
                    actions.remove(4)
                
                if action in actions:
                    return True
        return False

    def get_nets(self):
        trained_defend_net, trained_attack_net = None, None
        all_state = self.env.reset()
        if self.train_team:
            state = handle_obs(all_state, self.train_team)
            input_shape = len(state)

        if self.train_team == "attack":
            output_shape = self.env.attack_action_space_n
        if self.train_team == "defend":
            output_shape = self.env.defend_action_space_n
            

        if self.use_trained_defend_net:
            output_shape = self.env.defend_action_space_n
            trained_defend_net = DQN(input_shape, output_shape).to(self.device)
            trained_defend_net.eval()
            if self.defender_net_path:
                trained_defend_net.load_state_dict(torch.load(self.defender_net_path))

        if self.use_trained_attack_net:
            output_shape = self.env.attack_action_space_n
            trained_attack_net = DQN(input_shape, output_shape).to(self.device)
            trained_attack_net.eval()
            if self.attacker_net_path:
                trained_attack_net.load_state_dict(torch.load(self.attacker_net_path))

        return trained_defend_net, trained_attack_net

    def get_actions(self, all_state, trained_defend_net, trained_attack_net):
        if self.train_team:
            state = handle_obs(all_state, self.train_team)
            state_a = np.array(state, copy=False) # add an dimension for BATCH_SIZE!
            state_v = torch.tensor(state_a, dtype=torch.float).to(self.device)
        
        AI_state = []
        if self.train_team == 'attack':
            AI_state = handle_obs(all_state, 'defend')
        if self.train_team == 'defend':
            AI_state = handle_obs(all_state, 'attack')


        if len(AI_state) > 0:
            AI_state_a = np.array(AI_state, copy=False)
            AI_state_v = torch.tensor(AI_state_a, dtype=torch.float).to(self.device)
                
        actions = []
        trainer_action, AI_action = None, None
        all_actions = self.env.sample_actions()
        if self.train_team == 'attack':
            if self.use_trained_attack_net:
                trainer_q_values = trained_attack_net(state_v)
                sorted_q_values, index = trainer_q_values.sort(descending=True)
                trainer_actions = index.tolist()
                while self.check_action(trainer_actions[0], self.train_team) == False:
                    trainer_actions.pop(0)
                trainer_action = trainer_actions[0]

            else:
                for action_info in all_actions:
                    if action_info.team == 'attack':
                        action = action_info.action
                        action_team = action_info.team
                        while self.check_action(action, action_team) == False:
                            action = random.choice(list(range(7)))
                        trainer_action = action

            if self.use_trained_defend_net:
                AI_q_values = trained_defend_net(AI_state_v)
                sorted_q_values, index = AI_q_values.sort(descending=True)
                AI_actions = index.tolist()
                while self.check_action(AI_actions[0], self.AI_team) == False:
                    AI_actions.pop(0)
                AI_action = AI_actions[0]
            else:
                for action_info in all_actions:
                    if action_info.team == 'defend':
                        action = action_info.action
                        action_team = action_info.team
                        while self.check_action(action, action_team) == False:
                            action = random.choice(list(range(7)))
                        AI_action = action
            
            for action in all_actions:
                if action.team == self.train_team:
                    actions.append(trainer_action)
                else:
                    actions.append(AI_action)
        
        if self.train_team == 'defend':
            if self.use_trained_defend_net:
                trainer_q_values = trained_defend_net(state_v)
                sorted_q_values, index = trainer_q_values.sort(descending=True)
                trainer_actions = index.tolist()
                while self.check_action(trainer_actions[0], self.train_team) == False:
                    trainer_actions.pop(0)
                trainer_action = trainer_actions[0]
            else:
                for action_info in all_actions:
                    if action_info.team == 'defend':
                        action = action_info.action
                        action_team = action_info.team
                        while self.check_action(action, action_team) == False:
                            action = random.choice(list(range(7)))
                        trainer_action = action

            if self.use_trained_attack_net:
                AI_q_values = trained_attack_net(AI_state_v)
                sorted_q_values, index = AI_q_values.sort(descending=True)
                AI_actions = index.tolist()
                while self.check_action(AI_actions[0], self.AI_team) == False:
                    AI_actions.pop(0)
                AI_action = AI_actions[0]
            else:
                for action_info in all_actions:
                    if action_info.team == 'attack':
                        action = action_info.action
                        action_team = action_info.team
                        while self.check_action(action, action_team) == False:
                            action = random.choice(list(range(7)))
                        AI_action = action

            for action in all_actions:
                if action.team == self.train_team:
                    actions.append(trainer_action)
                else:
                    actions.append(AI_action)
        
        if len(actions) == 0:
            for action_info in all_actions:
                action = action_info.action
                action_team = action_info.team
                while self.check_action(action, action_team) == False:
                    action = random.choice(list(range(7)))
                actions.append(action)

        return actions, trainer_actions, AI_actions, trainer_action, AI_action, state_v