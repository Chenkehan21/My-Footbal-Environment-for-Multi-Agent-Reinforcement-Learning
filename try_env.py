from football_env import Football_Env
from Mywindow import mywindow
from panel_control import Panel_Control
from PyQt5 import QtWidgets
import sys
from utils import *


if __name__ == "__main__":
    app = 0
    app = QtWidgets.QApplication(sys.argv)
    env = Football_Env(agents_left=[1], agents_right=[2],
    max_episode_steps=500, move_reward_weight=1.0,
    court_width=23, court_height=20, gate_width=6)
    env.reset()
    for i in range(1):
        total_actions = env.sample_actions()
        actions = []
        for action in total_actions:
            actions.append(action.action)
        print("actions: ", actions)
        obs, rewards, done, info = env.step(actions)
        print(info)
    window = mywindow(6, env)
    control = Panel_Control(env, window)
    window.show()
    sys.exit(app.exec_())

    
    # env = Football_Env(agents_left=[1], agents_right=[2],
    # max_episode_steps=500, move_reward_weight=1.0,
    # court_width=23, court_height=20, gate_width=6)
    # env.reset()

    # for i in range(1):
    #     print(i)
    #     attack_total_reward = 0.0
    #     defend_total_reward = 0.0
    #     state = env.reset()
    #     print("state: ", state)
    #     while True:
    #         actions = env.sample_actions()
    #         obs, rewards, done, info = env.step(actions)
    #         print(obs)
    #         for reward in rewards:
    #             if reward.team == "attack":
    #                 attack_total_reward += reward.reward
    #             if reward.team == "defend":
    #                 defend_total_reward += reward.reward
    #         if done:
    #             print("****done****")
    #             print("attack total reward: ", attack_total_reward)
    #             print("defend total reward: ", defend_total_reward)
    #             print()
    #             break
        