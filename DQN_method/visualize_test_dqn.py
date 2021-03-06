import os
import sys
sys.path.append(os.path.dirname(__file__) + os.sep + '../')

import torch
from football_env import Football_Env
import numpy as np
from utils import *
from Mywindow import mywindow
from panel_control import Panel_Control
from PyQt5 import QtWidgets


if __name__ == "__main__":
    app = 0
    app = QtWidgets.QApplication(sys.argv)
    env = Football_Env(agents_left=[1, 2], agents_right=[3, 4],
    max_episode_steps=500, move_reward_weight=1.0,
    court_width=23, court_height=20, gate_width=6)
    env.reset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    defend_net_path1 = '/home/chenkehan/RESEARCH/codes/experiment4/DQN_method/res_defend4/football_0.860_98.000.dat'
    defend_net_path2 = '/home/chenkehan/RESEARCH/codes/experiment4/DQN_method/res_defend4/football_0.830_100.000.dat'
    attack_net_path1 = '/home/chenkehan/RESEARCH/codes/experiment4/DQN_method/res_attack5/football_0.710_59.780.dat'
    attack_net_path2 = '/home/chenkehan/RESEARCH/codes/experiment4/DQN_method/res_attack5/football_0.890_72.300.dat'
    # attack_net_path = '/home/chenkehan/RESEARCH/codes/experiment4/DQN_method/res_attack6/football_0.920_137.000.dat'
    window = mywindow(2, env)
    control = Panel_Control(env, train_team='attack', device=device,
                            use_trained_defend_net=True, use_trained_attack_net=True,
                            defend_net_path=[defend_net_path1, defend_net_path2], attack_net_path=[attack_net_path1, attack_net_path2],
                            window=window)
    window.show()
    sys.exit(app.exec_())