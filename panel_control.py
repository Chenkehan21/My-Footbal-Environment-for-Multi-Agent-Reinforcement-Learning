import os
import sys

from torch._C import dtype
sys.path.append(os.path.dirname(__file__) + os.sep + '../')

import torch
from utils import *
from DQN_model import DQN

PATH = '/home/chenkehan/RESEARCH/codes/experiment4/DQN_model/res_win_rate/football_0.880_66.500.dat'

class Panel_Control:
    def __init__(self, env, mywindow, train_team=None, device=None, 
                 use_trained_attack_net=False, use_trained_defend_net=False, 
                 attack_net_path=None, defend_net_path=None):
        self.env = env
        self.window = mywindow
        self.train_team = train_team
        self.device = device
        self.use_trained_attack_net = use_trained_attack_net
        self.use_trained_defend_net = use_trained_defend_net
        self.attack_net_path = attack_net_path
        self.defend_net_path = defend_net_path
        self.actionResponse()

    def doReset(self):
        print("Reset Button")
        self.env.reset()
        self.window.doRefresh(self.env)
    
    # def doStep(self):
    #     print("do step")
    #     while not self.env.Done:
    #         total_actions = self.env.sample_actions()
    #         actions = []
    #         for action in total_actions:
    #             actions.append(action.action)
    #         print("actions: ", actions)
    #         next_state, rewards, done, info = self.env.step(actions)
    #         self.window.doRefresh_Step(self.env)
    #         if done:
    #             print("game done winner: ", info['winner'])
    #             break

    def doStep(self):
        print("do step dqn, trained team: ", self.train_team)
        state = self.env.reset()
        state = handle_obs(state, self.train_team)

        if self.train_team == "attack":
            output_shape = self.env.attack_action_space_n
        if self.train_team == "defend":
            output_shape = self.env.defend_action_space_n

        input_shape = len(state)

        if self.use_trained_defend_net:
            output_shape = self.env.defend_action_space_n
            trained_defend_net = DQN(input_shape, output_shape).to(self.device)
            trained_defend_net.eval()
            if self.defend_net_path:
                trained_defend_net.load_state_dict(torch.load(self.defend_net_path))

        if self.use_trained_attack_net:
            output_shape = self.env.attack_action_space_n
            trained_attack_net = DQN(input_shape, output_shape).to(self.device)
            trained_attack_net.eval()
            if self.attack_net_path:
                trained_attack_net.load_state_dict(torch.load(self.attack_net_path))

        total_reward = 0.0
        all_state = self.env.reset()


        while not self.env.Done:
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
                    _, action_v = torch.max(trainer_q_values, dim=0)
                    trainer_action = int(action_v.item())
                else:
                    for action in all_actions:
                        if action.team == 'attack':
                            trainer_action = action.action

                if self.use_trained_defend_net:
                    AI_q_values = trained_defend_net(AI_state_v)
                    _, AI_action_v = torch.max(AI_q_values, dim=0)
                    AI_action = int(AI_action_v.item())
                    print("AI action: ", AI_action)
                else:
                    for action in all_actions:
                        if action.team == 'defend':
                            AI_action = action.action
                
                for action in all_actions:
                    if action.team == self.train_team:
                        actions.append(trainer_action)
                    else:
                        actions.append(AI_action)
            
            if self.train_team == 'defend':
                if self.use_trained_defend_net:
                    trainer_q_values = trained_defend_net(state_v)
                    _, action_v = torch.max(trainer_q_values, dim=0)
                    trainer_action = int(action_v.item())
                else:
                    for action in all_actions:
                        if action.team == 'defend':
                            trainer_action = action.action

                if self.use_trained_attack_net:
                    AI_q_values = trained_attack_net(AI_state_v)
                    _, AI_action_v = torch.max(AI_q_values, dim=0)
                    AI_action = int(AI_action_v.item())
                else:
                    for action in all_actions:
                        if action.team == 'attack':
                            AI_action = action.action

                for action in all_actions:
                    if action.team == self.train_team:
                        actions.append(trainer_action)
                    else:
                        actions.append(AI_action)
            
            if len(actions) == 0:
                for action in all_actions:
                    actions.append(action.action)
            
            # print("actions: ", actions)
            next_state, rewards, done, info = self.env.step(actions)

            for rew in rewards:
                if rew.team == self.train_team:
                    total_reward += rew.reward
            
            if done:
                print("game done winner: ", info['winner'])
                self.window.doRefresh_Step(self.env)
                break

            all_state = next_state
            self.window.doRefresh_Step(self.env)

    def doAuto(self):
        pass

    def actionResponse(self):
        self.window.Button_Reset.clicked.connect(self.doReset)
        self.window.Button_Step.clicked.connect(self.doStep)
        # self.window.Button_Auto.clicked.connect(self.doAuto)