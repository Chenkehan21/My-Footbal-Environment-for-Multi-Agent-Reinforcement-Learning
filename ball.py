import numpy as np


class Ball:
    def __init__(self):
        self.pos = [11, 10]
        self.blocked = False

    def give_ball_possession(self, pos):
        self.pos = pos

    def move(self, source, destination, _map): # the see_around() function in Player ensures that source != destination
        # need to be modefied!!
        self.blocked = False
        if destination[0] != source[0] and destination[1] != source[1]:
            k = (destination[1] - source[1]) / (destination[0] - source[0])
            b = source[1] - k * source[0]
            f = lambda x: k * x + b
            for x in range(source[0] + 1, destination[0]):
                y = f(x)
                if y % 1 == 0:
                    y = int(y)
                    if _map[x][y] != 0:
                        self.blocked = True
                        pos = [x, y]
            if not self.blocked:
                pos = destination
        elif destination[0] != source[0] and destination[1] == source[1]:
            start_id = min(source[0], destination[0]) + 1
            end_id = max(source[0], destination[0]) + 1
            for i in range(start_id, end_id):
                if _map[i][source[1]] != 0:
                    self.blocked = True
                    pos = [i, source[1]]
            if not self.blocked:
                pos = destination
        elif destination[0] == source[0] and destination[1] != source[1]:
            start_id = min(source[1], destination[1]) + 1
            end_id = max(source[1], destination[1]) + 1
            for i in range(start_id, end_id):
                if _map[source[0]][i] != 0:
                    self.blocked = True
                    pos = [source[0], i]
            if not self.blocked:
                pos = destination
        
        elif source == destination:
            pos = destination
        
        return pos

    def move2(self, source, destination, agents):
        self.blocked = False
        if destination[1] != source[1]:
            k = (destination[0] - source[0]) / (destination[1] - source[1])
            b = source[0] - k * source[1]
            f = lambda x: k * x + b
        elif destination[1] == source[1]:
            f = lambda x: source[1]
        elif destination[0] == source[0]:
            f = lambda x: source[0]
        
        for agent in agents.values():
            if agent.team == 'defend':
                defend_x = agent.pos[1]
                defend_y_min = agent.pos[0] - 1
                defend_y_max = agent.pos[0] + 1
                y = f(defend_x)
                if y >= defend_y_min and y <= defend_y_max and defend_x >= source[1] and defend_x <= destination[1]:
                    self.blocked = True
                    pos = [int(y), defend_x]
        if not self.blocked:
            pos = destination

        return pos

    # if players only pass ball to other players instead of passing randomly, check_ball_pass_court is useless.
    def check_ball_pass_court(self, _map):
        if self.pos[0] < 0 or self.pos[0] >= _map.shape[0] or self.pos[1] < 0 or self.pos[1] >= _map.shape[1]:
            return True
        else:
            return False

    def check_ball_score(self, team, court_id, court_width, court_height, gate_width=6):
        gate_range = [int(court_height / 2) - int(gate_width / 2), int(court_height / 2) + int(gate_width / 2) + 1]
        if team == "attack" and court_id == "left":
            if self.pos[1] == court_width - 1 and (self.pos[0] <= gate_range[1] and self.pos[0] >= gate_range[0]):
                return True
        if team == "attack" and court_id == "right":
            if self.pos[1] == 0 and (self.pos[0] <= gate_range[1] and self.pos[0] >= gate_range[0]):
                return True
        return False