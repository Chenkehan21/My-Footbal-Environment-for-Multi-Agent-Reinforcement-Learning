import os
import sys
import random
from copy import deepcopy
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

    
    def check_action(self, id, action):
        if self.env.agents[id].team == 'attack':
            # print("attack pos: ", [self.env.agents[id].pos[0], self.env.agents[id].pos[1]])
            actions = list(range(7))
            if self.env.agents[id].pos[0] == 0:
                # print("top")
                actions.remove(1)
            if self.env.agents[id].pos[0] == self.env.court_height - 1:
                # print("bottom")
                actions.remove(2)
            if self.env.agents[id].pos[1] == 0:
                # print("left")
                actions.remove(3)
            if self.env.agents[id].pos[1] == self.env.court_width - 1:
                # print("right")
                actions.remove(4)

            pass_pos, pass_agent_id = self.env.agents[id].can_pass(self.env._map, self.env.agents)
            if pass_pos == None:
                actions.remove(5)
            else:
                if self.env.agents[id].posses_ball == False:
                    actions.remove(5)

            shoot_pos = self.env.agents[id].can_shoot()
            if shoot_pos:
                shoot_success_rate = self.env.agents[id].success_rate[shoot_pos[0]]
                if (self.env.agents[id].posses_ball and shoot_pos and random.random() < shoot_success_rate) != True:
                    actions.remove(6)
            else:
                actions.remove(6)

        if self.env.agents[id].team == 'defend':
            # print("defend pos: ", [self.env.agents[id].pos[0], self.env.agents[id].pos[1]])
            actions = list(range(5))
            if self.env.agents[id].pos[0] == 0:
                # print("defend top")
                actions.remove(1)
            if self.env.agents[id].pos[0] == self.env.court_height - 1:
                # print("defend bottom")
                actions.remove(2)
            if self.env.agents[id].pos[1] == 0:
                # print("defend left")
                actions.remove(3)
            if self.env.agents[id].pos[1] == self.env.court_width - 1:
                # print("defend right")
                actions.remove(4)
        # print("proper actions: ", actions) 
        if action in actions:
            # print("ok action: ", action)
            return True

        return False

    def get_nets(self):
        trained_defend_nets, trained_attack_nets = None, None
        all_state = self.env.reset()
        state = handle_obs(all_state, self.train_team)
        input_shape = len(state[0])

        if self.train_team == "attack":
            output_shape = self.env.attack_action_space_n
        if self.train_team == "defend":
            output_shape = self.env.defend_action_space_n
            

        if self.use_trained_defend_net:
            output_shape = self.env.defend_action_space_n
            trained_defend_net = DQN(input_shape, output_shape).to(self.device)
            trained_defend_net.eval()
            trained_defend_nets = [trained_defend_net] * len(self.env.agents_right)
            if len(self.defender_net_path) > 0:
                for idx, net in enumerate(trained_defend_nets):
                    net.load_state_dict(torch.load(self.defender_net_path[idx]))

        if self.use_trained_attack_net:
            output_shape = self.env.attack_action_space_n
            trained_attack_net = DQN(input_shape, output_shape).to(self.device)
            trained_attack_net.eval()
            trained_attack_nets = [trained_attack_net] * len(self.env.agents_left)
            if self.attacker_net_path:
                for idx, net in enumerate(trained_attack_nets):
                    net.load_state_dict(torch.load(self.attacker_net_path[idx]))

        return trained_defend_nets, trained_attack_nets

    def get_actions(self, all_state, trained_defend_nets, trained_attack_nets):
        if self.train_team:
            states = handle_obs(all_state, self.train_team)
            state_a = [np.array(state, copy=False) for state in states] # add an dimension for BATCH_SIZE!
            state_v = [torch.tensor(state, dtype=torch.float).to(self.device) for state in state_a]
        
        AI_states = []
        if self.train_team == 'attack':
            AI_states = handle_obs(all_state, 'defend')
        if self.train_team == 'defend':
            AI_states = handle_obs(all_state, 'attack')


        if len(AI_states) > 0:
            AI_state_a = [np.array(AI_state, copy=False) for AI_state in AI_states]
            AI_state_v = [torch.tensor(state, dtype=torch.float).to(self.device) for state in AI_state_a]
                
        actions = []
        trainer_action_net, AI_action_net = [], []
        trainer_actions, AI_actions = [], []
        trainer_actions_list, AI_actions_list = [], []
        all_actions = self.env.sample_actions()
        if self.train_team == 'attack':
            if self.use_trained_attack_net:
                for idx, trained_attack_net in enumerate(trained_attack_nets):
                    trainer_q_values = trained_attack_net(state_v[idx])
                    sorted_q_values, index = trainer_q_values.sort(descending=True)
                    trainer_actions = index.tolist()
                    trainer_actions_list.append(trainer_actions)
                    while self.check_action(idx + 1, trainer_actions[0]) == False:
                        trainer_actions.pop(0)
                    trainer_action = trainer_actions[0]
                    trainer_action_net.append(trainer_action)
            else:
                for idx, action_info in enumerate(all_actions):
                    if action_info.team == 'attack':
                        action = action_info.action
                        action_team = action_info.team
                        while self.check_action(idx + 1, action) == False:
                            action = random.choice(list(range(7)))
                        trainer_action_net.append(action)

            if self.use_trained_defend_net:
                for idx, trained_defend_net in enumerate(trained_defend_nets):
                    AI_q_values = trained_defend_net(AI_state_v[idx])
                    sorted_q_values, index = AI_q_values.sort(descending=True)
                    AI_actions = index.tolist()
                    AI_actions_list.append(AI_actions)
                    while self.check_action(idx + len(self.env.agents_left) + 1, AI_actions[0]) == False:
                        AI_actions.pop(0)
                    AI_action = AI_actions[0]
                    AI_action_net.append(AI_action)
            else:
                for idx, action_info in enumerate(all_actions):
                    if action_info.team == 'defend':
                        action = action_info.action
                        action_team = action_info.team
                        while self.check_action(idx + 1, action) == False:
                            action = random.choice(list(range(7)))
                        AI_action_net.append(action)

            trainer_action_net2 = deepcopy(trainer_action_net)
            AI_action_net2 = deepcopy(AI_action_net)
            for action in all_actions:
                if action.team == self.train_team:
                    actions.append(trainer_action_net2[0])
                    trainer_action_net2.pop(0)
                else:
                    actions.append(AI_action_net2[0])
                    AI_action_net2.pop(0)
        
        if self.train_team == 'defend':
            if self.use_trained_defend_net:
                for idx, trained_defend_net in enumerate(trained_defend_nets):
                    trainer_q_values = trained_defend_net(state_v[idx])
                    sorted_q_values, index = trainer_q_values.sort(descending=True)
                    trainer_actions = index.tolist()
                    trainer_actions_list.append(trainer_actions)
                    while self.check_action(idx + len(self.env.agents_left) + 1, trainer_actions[0]) == False:
                        trainer_actions.pop(0)
                    trainer_action = trainer_actions[0]
                    trainer_action_net.append(trainer_action)
            else:
                for idx, action_info in enumerate(all_actions):
                    if action_info.team == 'defend':
                        action = action_info.action
                        action_team = action_info.team
                        while self.check_action(idx + 1, action) == False:
                            action = random.choice(list(range(7)))
                        trainer_action.append(action)

            if self.use_trained_attack_net:
                for idx, trained_attack_net in enumerate(trained_attack_nets):
                    AI_q_values = trained_attack_net(AI_state_v[idx])
                    sorted_q_values, index = AI_q_values.sort(descending=True)
                    AI_actions = index.tolist()
                    AI_actions_list.append(AI_actions)
                    while self.check_action(idx + 1, AI_actions[0]) == False:
                        AI_actions.pop(0)
                    AI_action = AI_actions[0]
                    AI_actions.append(AI_action)
            else:
                for idx, action_info in enumerate(all_actions):
                    if action_info.team == 'attack':
                        action = action_info.action
                        action_team = action_info.team
                        while self.check_action(idx + 1, action) == False:
                            action = random.choice(list(range(7)))
                        AI_action.append(action)

            trainer_action_net2 = deepcopy(trainer_action_net)
            AI_action_net2 = deepcopy(AI_action_net)
            for action in all_actions:
                if action.team == self.train_team:
                    actions.append(trainer_action_net2[0])
                    trainer_action_net2.pop(0)
                else:
                    actions.append(AI_action_net2[0])
                    AI_action_net2.pop(0)
        
        if len(actions) == 0:
            for idx, action_info in enumerate(all_actions):
                action = action_info.action
                action_team = action_info.team
                while self.check_action(idx + 1, action) == False:
                    action = random.choice(list(range(7)))
                actions.append(action)
        
        map_state = self.env._map

        return actions, trainer_actions_list, AI_actions_list, trainer_action_net, AI_action_net, state_v, map_state