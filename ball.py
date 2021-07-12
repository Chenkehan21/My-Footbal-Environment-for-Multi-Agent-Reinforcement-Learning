import numpy as np


class Ball:
    def __init__(self):
        self.pos = None
        self.blocked = False

    def give_ball_possession(self, pos):
        self.pos = pos

    def move(self, source, destination, _map): # the see_around() function in Player ensures that source != destination
        if destination[0] != source[0] and destination[1] != source[1]:
            if _map[destination[0]][destination[1]] != 0:
                self.blocked = True
            else:
                self.pos = destination
        if destination[0] != source[0] and destination[1] == source[1]:
            start_id = min(source[0], destination[0])
            end_id = max(source[0], destination[0]) + 1
            for i in range(start_id, end_id):
                if _map[i][source[1]] != 0:
                    self.blocked = True
        if destination[0] == source[0] and destination[1] != source[1]:
            start_id = min(source[1], destination[1])
            end_id = max(source[1], destination[1]) + 1
            for i in range(start_id, end_id):
                if _map[source[0]][i] != 0:
                    self.blocked = True

    # if players only pass ball to other players instead of passing randomly, check_ball_pass_court is useless.
    def check_ball_pass_court(self, _map):
        if self.pos[0] < 0 or self.pos[0] >= _map.shape[0] or self.pos[1] < 0 or self.pos[1] >= _map.shape[1]:
            return True
        else:
            return False

    def check_ball_score(self, team, court_id, court_width, court_height, gate_width=6):
        gate_range = [court_height / 2 - gate_width / 2, court_height / 2 + gate_width / 2]
        if team == "attack" and court_id == "left":
            if self.pos[0] == court_width and (self.pos[1] <= gate_range[1] and self.pos[1] >= gate_range[0]):
                return True
        if team == "attack" and court_id == "right":
            if self.pos[0] == 0 and (self.pos[1] <= gate_range[1] and self.pos[1] >= gate_range[0]):
                return True
        return False