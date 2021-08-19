import os
import sys
import random
from torch._C import dtype
sys.path.append(os.path.dirname(__file__) + os.sep + '../')

import torch
from utils import *
from DQN_method.agents import Agents


class Panel_Control(Agents):
    def __init__(self, env, train_team, device, 
                 use_trained_defend_net, use_trained_attack_net,
                 defend_net_path, attack_net_path, window):
        super(Panel_Control, self).__init__(
                env, train_team, device, 
                use_trained_defend_net, use_trained_attack_net,
                defend_net_path, attack_net_path
                )

        self.window = window
        self.trained_defend_net, self.trained_attack_net = self.get_nets()
        self.state = self.env.reset()
        self.actionResponse()

    def doReset(self):
        print("\nReset Button")
        self.state = self.env.reset()
        self.window.doRefresh(self.env)
    
    def doRandomStep(self):
        if not self.env.Done:
            total_actions = self.env.sample_actions()
            actions = []
            for action in total_actions:
                actions.append(action.action)
            next_state, rewards, done, info = self.env.step(actions)
            self.window.doRefresh_Step(self.env)
            if done:
                self.window.doRefresh_Step(self.env)
                print("game done winner: ", info['winner'], "\n")
    
    def doRandomAuto(self):
        while not self.env.Done:
            total_actions = self.env.sample_actions()
            actions = []
            for action in total_actions:
                actions.append(action.action)
            next_state, rewards, done, info = self.env.step(actions)
            self.window.doRefresh_Step(self.env)
            if done:
                self.window.doRefresh_Step(self.env)
                print("game done winner: ", info['winner'], "\n")
                break

    def doAIStep(self):
        if not self.env.Done:
            actions, _, _, _, _, _, _ = self.get_actions(self.state, self.trained_defend_net, self.trained_attack_net)
            next_state, rewards, done, info = self.env.step(actions)

            if done:
                print("game done winner: ", info['winner'], "\n")
                self.window.doRefresh_Step(self.env)

            self.state = next_state
            self.window.doRefresh_Step(self.env)
    
    def doAIAuto(self):
        all_state = self.env.reset()

        while not self.env.Done:
            actions, _, _, _, _, _, _ = self.get_actions(all_state, self.trained_defend_net, self.trained_attack_net)
            next_state, rewards, done, info = self.env.step(actions)

            if done:
                print("game done winner: ", info['winner'], "\n")
                self.window.doRefresh_Step(self.env)
                break

            all_state = next_state
            self.window.doRefresh_Step(self.env)

    def actionResponse(self):
        self.window.Button_Reset.clicked.connect(self.doReset)
        self.window.Button_Random_Step.clicked.connect(self.doRandomStep)
        self.window.Button_Random_Auto.clicked.connect(self.doRandomAuto)
        if self.use_trained_defend_net or self.use_trained_attack_net:
            self.window.Button_AI_Step.clicked.connect(self.doAIStep)
            self.window.Button_AI_Auto.clicked.connect(self.doAIAuto)