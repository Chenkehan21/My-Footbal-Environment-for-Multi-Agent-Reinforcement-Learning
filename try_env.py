from football_env import Football_Env
from Mywindow import mywindow
from panel_control import Panel_Control
from PyQt5 import QtWidgets
import sys
import torch
from utils import *


if __name__ == "__main__":
    app = 0
    app = QtWidgets.QApplication(sys.argv)
    env = Football_Env(agents_left=[1,2,], agents_right=[3, 4],
    max_episode_steps=500, move_reward_weight=1.0,
    court_width=23, court_height=20, gate_width=6)
    env.reset()
    window = mywindow(4, env)
    control = Panel_Control(env, train_team='attack', device=torch.device("cpu"),
                            use_trained_defend_net=False, use_trained_attack_net=False,
                            defend_net_path=[], attack_net_path=[],
                            window=window)
    window.show()
    sys.exit(app.exec_())