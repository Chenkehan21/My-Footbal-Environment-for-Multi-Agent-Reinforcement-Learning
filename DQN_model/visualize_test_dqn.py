import os
import sys
sys.path.append(os.path.dirname(__file__) + os.sep + '../')

import torch
from football_env import Football_Env
import numpy as np
from utils import *
from DQN_model import DQN
from Mywindow import mywindow
from panel_control import Panel_Control
from PyQt5 import QtWidgets


if __name__ == "__main__":
    app = 0
    app = QtWidgets.QApplication(sys.argv)
    env = Football_Env(agents_left=[1], agents_right=[2],
    max_episode_steps=500, move_reward_weight=1.0,
    court_width=23, court_height=20, gate_width=6)
    env.reset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    defend_net_path = '/home/chenkehan/RESEARCH/codes/experiment4/DQN_model/res_win_rate/football_0.880_66.500.dat'
    attack_net_path = '/home/chenkehan/RESEARCH/codes/experiment4/DQN_model/res_attack/football_29.240_0.780.dat'
    window = mywindow(2, env)
    control = Panel_Control(env, window, train_team='attack', device=device, 
                            defend_net_path=defend_net_path, attack_net_path=attack_net_path,
                            use_trained_defend_net=True, use_trained_attack_net=True)
    window.show()
    sys.exit(app.exec_())