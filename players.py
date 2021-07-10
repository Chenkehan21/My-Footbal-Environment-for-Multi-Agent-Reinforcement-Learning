import numpy as np
import random
import itertools
import math

class Players:
    '''
    attackers' action space:
    0: None
    1: Up
    2: Down
    3: Left
    4: Right
    5: Pass
    6: Shoot

    defenders' action space:
    0: None
    1: Up
    2: Down
    3: Left
    4: Right
    '''
    def __init__(self, agent_id, court_id, team, court_width, court_height):
        self.id = agent_id # start from 1
        self.court_id = court_id # court_id is left or right
        self.team = team # team can be attack or defender
        self.court_width = court_width
        self.court_height = court_height
        self.pos = None
        self.reset_position(_map)
        self.actions = []
        self.gate_pos = self.get_gate_pos()
        self.posses_ball = None

    def _get_obs(self, _map):
        my_pos = self.pos
        around_obs = self.see_around(_map)
        gate_pos = self.gate_pos

        obs = np.concatenate(
            [np.array(my_pos), 
            np.array(around_obs), 
            np.array(gate_pos)],
            np.array([self.posses_ball])
        )
        return obs

    def sample_action(self):
        if self.team == "attack":
            action = random.choice(list(range(7)))
        if self.team == "deffend":
            action = random.choice(list(range(5)))
        return action

    def reset_position(self):
        left_court_start = 0
        left_court_end = math.floor(self.court_width / 2) + 1
        right_court_start = math.ceil(self.court_width / 2)
        right_court_end = self.court_width

        if self.court_id == "left":
            x = random.choice(list(range(left_court_start, left_court_end))) # top left conner is (0, 0)
            y = random.choice(list(range(0, self.court_height)))
            if _map[x][y] != 0:
                self.reset_position(_map)
        if self.court_id == "right":
            x = random.choice(list(range(right_court_start, right_court_end))) # top left conner is (0, 0)
            y = random.choice(list(range(0, court_height)))
            if _map[x][y] != 0:
                self.reset_position(_map)
        self.pos = [x, y]

    def before_step(self):
        pass

    def see_around(self, _map): # map is a two dimension. "map[i][j] = x" means agent_x at position (i, j)
        agents_around = []
        court_width = _map.shape[1]
        court_height = _map.shape[0]
        row_start = self.pos[1] - 1 if self.pos[1] - 1 >= 0 else 0
        row_end = self.pos[1] + 2 if self.pos[1] + 2 <= court_height else court_height
        col_start = self.pos[0] - 1 if self.pos[0] - 1 >= 0 else 0
        col_end = self.pos[0] + 2 if self.pos[0] + 2 <= court_width else court_width
        for i in range(row_start, row_end):
            for j in range(col_start, col_end):
                if _map[i][j] != 0 and _map[i][j] != self.id:
                    agents_around.append([i, j])
        return agents_around

    def get_gate_pos(self):
        pass

    def can_shoot(self):
        pass

    def check_agents_pass_court(self):
        pass

    def check_agents_conflict(self, agent_pos, _map):
        if _map[agent_pos[0]][agent_pos[1]] != 0:
            return True
        return False
    
    def simulate_move(self, action, _map, ball):
        if action == 0:
            virtual_agent_pos = self.pos
        if action == 1:
            virtual_agent_pos = [self.pos[0], self.pos[1] - 1]
        if action == 2:
            virtual_agent_pos = [self.pos[0], self.pos[1] + 1]
        if action == 3:
            virtual_agent_pos = [self.pos[0] - 1, self.pos[1]]
        if action == 4:
            virtual_agent_pos = [self.pos[0] + 1, self.pos[1]]
        if self.team == "attack":
            agents_around = self.see_around(_map)
            if agents_around:
                pass_agent_pos = random.choice(agents_around)
                if action == 5 and self.posses_ball:
                    virtual_ball_pos = ball.move(self.pos, pass_agent_pos, _map) # if the ball runs into boundary or defenders it will be stoped and one episode done
            if action == 6 and self.can_shoot():
                virtual_ball_pos = ball.move(self.pos, self.gate_pos, _map)
            virtual_agent_pos = self.pos
        return virtual_agent_pos, virtual_ball_pos

    def _step(self, action, _map, ball):
        done = False
        penalty = 0.0
        done_reward = 0.0
        move_reward = 0.0
        virtual_agent_pos, virtual_ball_pos = self.simulate_move(action, _map, ball)

        '''done reward
        '''
        self.actions.append(action)
        action_records = {}
        for k, v, in itertools.groupby(self.actions):
            action_records[k] = list(v)
        if len(action_records[0]) == 5: # if not move more than 3 steps give a minus reward
            penalty = -1
            self.actions.clear()
        if len(self.actions) > 1000:
            self.actions.clear() # avoid memory leak
        if self.check_agents_pass_court(virtual_agent_pos):
            agent_pass_court = True
            done_reward = -1
        if ball.check_ball_pass_court(virtual_ball_pos):
            ball_passs_court = True
            done_reward = -1
        if ball.check_ball_score(virtual_ball_pos, self.team, self.court_id, _map):
            score = True
            done_reward = 1
        if ball.check_ball_score(virtual_ball_pos, self.team, self.court_id, _map):
            score = True
            done_reward = -1

        if agent_pass_court or ball_passs_court or score or self.check_agents_conflict():
            done = True

        if not done:
            '''move reward
            '''
            if self.team == "attack":
                if self.court_id == "left" and action == 4:
                    move_reward = 1.0
                if self.court_id == "left" and action == 3:
                    move_reward = -1.0
                if self.court_id == "right" and action == 3:
                    move_reward = 1.0
                if self.court_id == "right" and action == 4:
                    move_reward = -1.0
            else:
                if self.pos[1] - ball.pos[1] < 0 and action == 2:
                    move_reward = 1.0
                if  self.pos[1] - ball.pos[1] < 0 and action == 1:
                    move_reward = -1.0
                if self.pos[1] - ball.pos[1] > 0 and action == 1:
                    move_reward = 1.0
                if self.pos[1] - ball.pos[1] > 0 and action == 2:
                    move_reward = -1.0
                if self.pos[1] - ball.pos[1] == 0 and (action == 1 or action ==2):
                    move_reward = -1.0

        reward = done_reward + move_reward + penalty        
        return reward, done