from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtCore import Qt
import os
import sys

class UI_FORM:
    def setup_UI(self, Form):
        Form.setObjectName("Form")
        Form.resize(1000, 700)
        Form.setMinimumSize(QtCore.QSize(500, 550))
        Form.setMaximumSize(QtCore.QSize(500, 550))

        self.Football_court = QtWidgets.QGroupBox(Form)
        self.Football_court.setGeometry(QtCore.QRect(0, 0, 500, 475))
        self.Football_court.setStyleSheet("")
        self.Football_court.setObjectName("Football_court")

        self.Automatic_Control = QtWidgets.QGroupBox(Form)
        self.Automatic_Control.setGeometry(QtCore.QRect(0, 475, 500, 225))
        self.Automatic_Control.setObjectName("Automatic_Control")
        
        self.Button_Reset = QtWidgets.QPushButton(self.Automatic_Control)
        self.Button_Reset.setGeometry(QtCore.QRect(10, 20, 75, 23))
        self.Button_Reset.setIconSize(QtCore.QSize(16, 16))
        self.Button_Reset.setObjectName("Button_Reset")
        
        self.Button_Random_Step = QtWidgets.QPushButton(self.Automatic_Control)
        self.Button_Random_Step.setGeometry(QtCore.QRect(95, 20, 105, 23))
        self.Button_Random_Step.setObjectName("Button_Step")

        self.Button_AI_Step = QtWidgets.QPushButton(self.Automatic_Control)
        self.Button_AI_Step.setGeometry(QtCore.QRect(210, 20, 75, 23))
        self.Button_AI_Step.setObjectName("Button_Step")

        self.Button_Random_Auto = QtWidgets.QPushButton(self.Automatic_Control)
        self.Button_Random_Auto.setGeometry(QtCore.QRect(300, 20, 105, 23))
        self.Button_Random_Auto.setObjectName("Button_Auto")

        self.Button_AI_Auto = QtWidgets.QPushButton(self.Automatic_Control)
        self.Button_AI_Auto.setGeometry(QtCore.QRect(420, 20, 75, 23))
        self.Button_AI_Auto.setObjectName("Button_Auto")
        
        # self.Experiment_Settings = QtWidgets.QGroupBox(self.Automatic_Control)
        # self.Experiment_Settings.setGeometry(QtCore.QRect(10, 60, 440, 80))
        # self.Experiment_Settings.setObjectName("Experiment_Settings")
        # self.ExperimentSetting = QtWidgets.QGridLayout(self.Experiment_Settings)
        # self.ExperimentSetting.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        # self.ExperimentSetting.setObjectName("ExperimentSetting")
        
        # self.E1 = QtWidgets.QLabel(self.Experiment_Settings)
        # self.E1.setObjectName("E1")
        # self.ExperimentSetting.addWidget(self.E1, 0, 0, 1, 1)
        
        # self.E3 = QtWidgets.QLabel(self.Experiment_Settings)
        # self.E3.setObjectName("E3")
        # self.ExperimentSetting.addWidget(self.E3, 0, 2, 1, 1)
        
        # self.Agent_num = QtWidgets.QLineEdit(self.Experiment_Settings)
        # self.Agent_num.setObjectName("Agent_num")
        # self.ExperimentSetting.addWidget(self.Agent_num, 0, 1, 1, 1)
        
        # self.Steps = QtWidgets.QLineEdit(self.Experiment_Settings)
        # self.Steps.setObjectName("Steps")
        # self.ExperimentSetting.addWidget(self.Steps, 0, 3, 1, 1)
        
        self.retranslate_UI(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslate_UI(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.Football_court.setTitle(_translate("Form", "Football Court"))

        self.Button_Reset.setText(_translate("Form", "Reset"))
        self.Button_Random_Step.setText(_translate("Form", "Random Step"))
        self.Button_AI_Step.setText(_translate("Form", "AI Step"))
        self.Button_Random_Auto.setText(_translate("Form", "Random Auto"))
        self.Button_AI_Auto.setText(_translate("Form", "AI Auto"))

        # self.Experiment_Settings.setTitle(_translate("Form", "Experiment Settings"))

        # self.E1.setText(_translate("Form", "Agent:"))
        # self.E3.setText(_translate("Form", "Steps:"))

        # self.Mannual_Control.setTitle(_translate("Form", "Mannual Control"))
        # self.Button_Left.setText(_translate("Form", "Left"))
        # self.BUtton_Ahead.setText(_translate("Form", "Ahead"))
        # self.Button_Right.setText(_translate("Form", "Right"))
        # self.Button_RightArrow.setText(_translate("Form", "..."))
    
o_path = os.getcwd()
sys.path.append(o_path + '/images')