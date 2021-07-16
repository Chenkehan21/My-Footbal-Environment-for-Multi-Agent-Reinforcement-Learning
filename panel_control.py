from football_env import Football_Env
from Mywindow import mywindow
from UI_form import UI_FORM

class Panel_Control:
    def __init__(self, env, mywindow):
        self.env = env
        self.window = mywindow
        self.actionResponse()

    def doReset(self):
        print("Reset Button")
        self.env.reset()
        self.window.doRefresh(self.env)
    
    def doStep(self):
        if not self.env.Done:
            actions = self.env.sample_actions()
            self.env.step(actions)
            self.window.doRefresh_Step(self.env)
        else:
            print("env done!!")

    
    def doAuto(self):
        pass

    def actionResponse(self):
        self.window.Button_Reset.clicked.connect(self.doReset)
        self.window.Button_Step.clicked.connect(self.doStep)
        # self.window.Button_Auto.clicked.connect(self.doAuto)

