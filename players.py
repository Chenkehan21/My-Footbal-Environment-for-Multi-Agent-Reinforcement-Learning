import numpy as np
import random
import itertools
import math

class Players:
    '''
    attackers' action space:
    0: Stand still
    1: Up
    2: Down
    3: Left
    4: Right
    5: Pass
    6: Shoot

    defenders' action space:
    0: Stand still
    1: Up
    2: Down
    3: Left
    4: Right
    '''

    def __init__(self, agent_id, court_id, team, court_width, court_height, gate_width, _map):
        self.id = agent_id # start from 1
        self.court_id = court_id # court_id is left or right
        self.team = team # team can be attacker or defender
        self.court_width = court_width
        self.court_height = court_height
        self.gate_width = gate_width
        self.pos = None
        self._map = _map
        self.actions = []
        self.gate_pos = self.get_gate_pos()
        self.posses_ball = False

    def _get_obs(self, _map):
        my_pos = self.pos
        around_obs, _ = self.see_around(_map)
        gate_pos = self.gate_pos
        # print(my_pos)
        # print(around_obs)
        # print(gate_pos)
        gate_pos = np.squeeze(np.array(gate_pos).reshape(1, -1))
        obs = np.concatenate(
            [np.array(my_pos), 
            np.array(around_obs), 
            np.array(gate_pos),
            np.array([self.posses_ball])]
        )
        
        return obs

    def sample_action(self):
        if self.team == "attack":
            action = random.choice(list(range(7)))
        if self.team == "defend":
            action = random.choice(list(range(5)))
        
        return action

    def see_around(self, _map): # map is a two dimension. "map[i][j] = x" means agent_x at position (i, j)
        agents_around = []
        agents_pos_around = []
        court_width = _map.shape[1]
        court_height = _map.shape[0]
        row_start = self.pos[1] - 1 if self.pos[1] - 1 >= 0 else 0
        row_end = self.pos[1] + 2 if self.pos[1] + 2 <= court_height else court_height
        col_start = self.pos[0] - 1 if self.pos[0] - 1 >= 0 else 0
        col_end = self.pos[0] + 2 if self.pos[0] + 2 <= court_width else court_width
        for i in range(row_start, row_end):
            for j in range(col_start, col_end):
                agents_around.append(_map[i][j])
                if _map[i][j] != 0 and _map[i][j] != self.id:
                    agents_around.append([i, j])

        
        return agents_around, agents_pos_around

    def get_gate_pos(self):
        left_gate_pos = [
                [i, 0] for i in 
                range(
                    int(self.court_height / 2) - int(self.gate_width / 2), 
                    int(self.court_height / 2) + int(self.gate_width / 2) + 1
                    )
            ]
        right_gate_pos = [
                [i, self.court_width - 1] for i in 
                range(
                    int(self.court_height / 2) - int(self.gate_width / 2), 
                    int(self.court_height / 2) + int(self.gate_width / 2) + 1
                    )
            ]
        if self.court_id == "left":
            my_gate_pos = left_gate_pos
            attack_gate_pos = right_gate_pos
        if self.court_id == "right":
            my_gate_pos = right_gate_pos
            attack_gate_pos = left_gate_pos
        
        return [my_gate_pos, attack_gate_pos]
    
    def simulate_move(self, action, _map, ball, agents):
        # print("action in simulate: ", action)
        pass_blocked, shoot_blocked = False, False
        virtual_ball_pos = ball.pos
        virtual_agent_pos = self.pos
        if action == 0: # stand still
            virtual_agent_pos = self.pos
        if action == 1: # UP
            virtual_agent_pos = [self.pos[0] - 1, self.pos[1]]
        if action == 2: # Down
            virtual_agent_pos = [self.pos[0] + 1, self.pos[1]]
        if action == 3: # Left
            virtual_agent_pos = [self.pos[0], self.pos[1] - 1]
        if action == 4: # Right
            virtual_agent_pos = [self.pos[0], self.pos[1] + 1]
        if self.team == "attack":
            _, agents_pos_around = self.see_around(_map)
            if agents_pos_around:
                pass_agent_pos = random.choice(agents_pos_around)
                if action == 5 and self.posses_ball: # Pass
                    # if the ball runs into boundary or defenders it will be stoped and one episode done
                    # print("pass")
                    ball.move(self.pos, pass_agent_pos, _map)
                    virtual_ball_pos = ball.pos
                    if ball.blocked:
                        block_agent_id = _map[ball.pos[0]][ball.pos[1]]
                        if agents[block_agent_id].team == "defend":
                            pass_blocked = True
            if action == 6: # Shoot
                # print("shoot")
                attack_gate_pos = self.gate_pos[1]
                attack_pos = random.choice(attack_gate_pos)
                # print("attack pos: ", attack_pos)
                ball.move(self.pos, attack_pos, _map)
                virtual_ball_pos = ball.pos
                if ball.blocked:
                    block_agent_id = _map[ball.pos[0]][ball.pos[1]]
                    # print("in players, agent team: ", agents[block_agent_id].team)
                    if agents[block_agent_id].team == "defend":
                        shoot_blocked = True
            if action == 5 or action == 6:
                virtual_agent_pos = self.pos
            # print("virtual ball pos: ", virtual_ball_pos)
        # print("virtual agent pos: ", virtual_agent_pos)
        # print("virtual ball pos: ", virtual_ball_pos)
        # print()

        return virtual_agent_pos, virtual_ball_pos, shoot_blocked, pass_blocked

    def after_step(self, action, _map, ball, agents):
        stand_still_penalty = 0.0
        attack_reward, defend_reward, block_reward = 0.0, 0.0, 0.0
        reward_info = dict()
        # print("simulate in after step: ")
        _, _, shoot_blocked, pass_blocked = self.simulate_move(action, _map, ball, agents)

        # check action = 0, we don't want the agent always stand still
        self.actions.append(action)
        action_records = {}
        for k, v, in itertools.groupby(self.actions):
            action_records[k] = list(v)
        if 0 in action_records.keys() and len(action_records[0]) >= 5:
            # if not move more than 3 steps give a penalty
            stand_still_penalty = -1.0
            self.actions.clear()
        if len(self.actions) > 1000:
            self.actions.clear() # avoid memory leak

        if self.team == "attack":
            if self.court_id == "left" and action == 4:
                attack_reward = 1.0
            if self.court_id == "left" and action == 3:
                attack_reward = -1.0
            if self.court_id == "right" and action == 3:
                attack_reward = 1.0
            if self.court_id == "right" and action == 4:
                attack_reward = -1.0

        if self.team == "defend":
            if self.pos[0] - ball.pos[0] < 0 and action == 2:
                defend_reward = 1.0
            if  self.pos[0] - ball.pos[0] < 0 and action == 1:
                defend_reward = -1.0
            if self.pos[0] - ball.pos[0] > 0 and action == 1:
                defend_reward = 1.0
            if self.pos[0] - ball.pos[0] > 0 and action == 2:
                defend_reward = -1.0
            if self.pos[0] - ball.pos[0] == 0 and (action == 1 or action ==2):
                defend_reward = -1.0

            if pass_blocked or shoot_blocked:
                block_reward = 1.0

        reward = attack_reward + defend_reward + block_reward + stand_still_penalty
        reward_info["attack"] = attack_reward
        reward_info["defend"] = defend_reward
        reward_info["block"] = block_reward
        reward_info["stand_still_penalty"] = stand_still_penalty
        
        return reward, reward_info