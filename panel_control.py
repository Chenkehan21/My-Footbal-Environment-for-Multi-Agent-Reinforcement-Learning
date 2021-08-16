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
        # print("path: ", self.defender_net_path)
        self.trained_defend_net, self.trained_attack_net = self.get_nets()
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
        print("do step, trained team: ", self.train_team)
        all_state = self.env.reset()

        while not self.env.Done:
            actions, _, _, _, _, _, _ = self.get_actions(all_state, self.trained_defend_net, self.trained_attack_net)
            # print("actions: ", actions)
            next_state, rewards, done, info = self.env.step(actions)

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