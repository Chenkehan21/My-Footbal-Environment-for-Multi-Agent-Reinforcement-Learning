from football_env import Football_Env
from Mywindow import mywindow
from panel_control import Panel_Control
from PyQt5 import QtWidgets
import sys


if __name__ == "__main__":
    app = 0
    app = QtWidgets.QApplication(sys.argv)
    env = Football_Env(agents_left=[1,2,3], agents_right=[4,5,6],
    max_episode_steps=500, move_reward_weight=1.0,
    court_width=23, court_height=20, gate_width=6)
    env.reset()
    for i in range(1):
        actions = env.sample_actions()
        obs, rewards, done, info = env.step(actions)
    window = mywindow(1, env)
    control = Panel_Control(env, window)
    window.show()
    sys.exit(app.exec_())

    
    # env = Football_Env(agents_left=[1,2,3], agents_right=[4,5,6],
    # max_episode_steps=500, move_reward_weight=1.0,
    # court_width=23, court_height=20, gate_width=6)
    # env.reset()

    # for i in range(10000):
    #     print(i)
    #     env.reset()
    #     while True:
    #         actions = env.sample_actions()
    #         obs, reward, done, info = env.step(actions)
    #         if done:
    #             # print("=======done=======")
    #             break
        