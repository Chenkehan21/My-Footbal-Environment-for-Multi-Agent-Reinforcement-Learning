from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QDockWidget, QListWidget
from PyQt5.QtGui import *

from UI_form import UI_FORM
from football_env import Football_Env

import os
import sys

class mywindow(QtWidgets.QMainWindow, UI_FORM):
    def __init__(self, agt_num: int, env):
    #这里需要重载一下mywindow，同时也包含了QtWidgets.QMainWindow的预加载项。
        super(mywindow, self).__init__()
        self.MAXSTEP = 500
        self.Icon_x = 20
        self.Icon_y = 20
        self.graphic = True
        self.setup_UI(self)
        self.__ball = [0, 0]
        self.__agent_num = agt_num
        self.init_Panel(env) #多个智能体面板的时候这里就要修改
        self.doRefresh(env)

    def init_Panel(self, env): #这个mode参数暂时没用 mode = true: snoar || false: avsonar
        # Parameter of Maze Panel
        self.__numStep = [0] * self.__agent_num
        self.__all_agents_pos = env.get_all_pos()

        self.loadImageIcons(env)
        self.draw_Panel(env)
    
    def loadImageIcons(self, env):
        self.L_background = QtWidgets.QLabel(self.Football_court)
        # self.L_gate = [QtWidgets.QLabel(self.Football_court) for i in range(env.numMines)]
        self.L_ball = QtWidgets.QLabel(self.Football_court)
        self.L_player1 = [QtWidgets.QLabel(self.Football_court) for i in range(len(env.agents_left))]
        self.L_player2 = [QtWidgets.QLabel(self.Football_court) for i in range(len(env.agents_right))]

        # Load Minefield
        self.Background = QtGui.QPixmap('./images/football_court.jpg')
        self.Ballicon = QtGui.QPixmap('./images/football.png')
        # self.Playericon = QtGui.QPixmap('./images/bomb.png')
        
        self.Player1icon = []
        self.Player2icon = []

        self.Player1icon.append(QtGui.QPixmap('./images/man1.png'))
        self.Player2icon.append(QtGui.QPixmap('./images/man2.png'))

    def doRefresh(self, env): #配合doReset
        self.__ball = env.ball.pos
        self.__all_agents_pos = self.setCurrent(env)
        self.update()

        self.__numStep = [0] * self.__agent_num
        if(self.graphic):
            self.RePaint(env)

    def setCurrent(self, env):
        return env.get_all_pos()

    def draw_Panel(self, env):
        #    Paint Minefield-Background
        self.L_background.setObjectName('L_background')
        self.L_background.setGeometry(QtCore.QRect(0, 0, 500, 450))
        self.L_background.setPixmap(self.Background)
        self.L_background.setScaledContents(True)

        # Paint football
        self.L_ball.setObjectName('L_ball')
        self.L_ball.setGeometry(QtCore.QRect(self.__ball[1]*self.Icon_x, self.__ball[0]*self.Icon_y, self.Icon_x, self.Icon_y))
        self.L_ball.setPixmap(self.Ballicon)
        self.L_ball.setScaledContents(True)

        # Paint players
        for index, id in enumerate(env.agents_left):
            self.L_player1[index].setObjectName('L_player1'+str(id))
            self.L_player1[index].setGeometry(QtCore.QRect(self.__all_agents_pos[id][1] * self.Icon_x, self.__all_agents_pos[id][0] * self.Icon_y, self.Icon_x, self.Icon_y))
            self.L_player1[index].setPixmap(self.Player1icon[0])
            self.L_player1[index].setScaledContents(True)
        
        for index, id in enumerate(env.agents_right):
            self.L_player2[index].setObjectName('L_player2'+str(id))
            self.L_player2[index].setGeometry(QtCore.QRect(self.__all_agents_pos[id][1] * self.Icon_x, self.__all_agents_pos[id][0] * self.Icon_y, self.Icon_x, self.Icon_y))
            self.L_player2[index].setPixmap(self.Player2icon[0])
            self.L_player2[index].setScaledContents(True)

    # 重绘事件
    def RePaint(self, env):
        # Paint football
        self.L_ball.setGeometry(QtCore.QRect(self.__ball[1]*self.Icon_x+25, self.__ball[0]*self.Icon_y+30, self.Icon_x*0.5, self.Icon_y*0.5))

        # Paint players
        for index, id in enumerate(env.agents_left):
            self.L_player1[index].setGeometry(QtCore.QRect(self.__all_agents_pos[id][1]*self.Icon_x, self.__all_agents_pos[id][0]*self.Icon_y, self.Icon_x*2, self.Icon_y*2))
            self.L_player1[index].setPixmap(self.Player1icon[0])

        for index, id in enumerate(env.agents_right):
            self.L_player2[index].setGeometry(QtCore.QRect(self.__all_agents_pos[id][1]*self.Icon_x, self.__all_agents_pos[id][0]*self.Icon_y, self.Icon_x*2, self.Icon_y*2))
            self.L_player2[index].setPixmap(self.Player2icon[0])

        # 设置信息面板为空, 刷新面板
        QtWidgets.QApplication.processEvents()

        if(self.graphic):
            self.update()

if __name__ == '__main__': #如果整个程序是主程序
    # QApplication相当于main函数， 也就是整个程序（很多文件）的主入口函数。
    # 对于GUI程序必须至少有一个这样的实例来让程序运行。
    app = 0
    app = QtWidgets.QApplication(sys.argv)
    #生成 mywindow 类的实例。
    env = Football_Env(agents_left=[1,2,3], agents_right=[4,5,6],
    max_episode_steps=500, move_reward_weight=1.0,
    court_width=23, court_height=20, gate_width=6)
    env.reset()
    print("attack court: ", env.attack_court)
    print("ball pos: ", env.ball.pos)
    window = mywindow(1, env)
    #有了实例，就得让它显示，show()是QWidget的方法，用于显示窗口。
    window.show()
    # 调用sys库的exit退出方法，条件是app.exec_()，也就是整个窗口关闭。
    # 有时候退出程序后，sys.exit(app.exec_())会报错，改用app.exec_()就没事
    # https://stackoverflow.com/questions/25719524/difference-between-sys-exitapp-exec-and-app-exec
    sys.exit(app.exec_())