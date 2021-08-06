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
        self.log = True
        self.miss_prob = 0.4

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
        row_start = self.pos[0] - 1 if self.pos[0] - 1 >= 0 else 0
        row_end = self.pos[0] + 2 if self.pos[0] + 2 <= court_height else court_height
        col_start = self.pos[1] - 1 if self.pos[1] - 1 >= 0 else 0
        col_end = self.pos[1] + 2 if self.pos[1] + 2 <= court_width else court_width
        for i in range(row_start, row_end):
            for j in range(col_start, col_end):
                agents_around.append(_map[i][j])
                if _map[i][j] != 0 and _map[i][j] != self.id:
                    agents_pos_around.append([i, j])

        while len(agents_around) != 9:
            agents_around.append(0)

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

    def can_shoot(self):
        pos_condition = False
        gate_condition = False
        shoot_range = [self.pos[0] - 1, self.pos[0], self.pos[0] + 1]

        if self.team == 'attack':
            if self.court_id == 'left' and self.pos[1] > self.court_width / 2:
                pos_condition = True
            elif self.court_id == 'right' and self.pos[1] < self.court_width / 2:
                pos_condition = True
            attack_gate_pos = self.gate_pos[1]
            attack_gate_range = [x[0] for x in attack_gate_pos]
            for x in shoot_range:
                if x in attack_gate_range:
                    gate_condition = True
                    shoot_pos = [x, attack_gate_pos[0][1]]
        
        if pos_condition and gate_condition:
            return shoot_pos
        else:
            return None
            
    
    def simulate_move(self, action, _map, ball, agents):
        # print("action in simulate: ", action)
        done = False
        pass_blocked, shoot_blocked = False, False
        virtual_ball_pos = ball.pos
        virtual_agent_pos = self.pos
        if action == 0: # stand still
            virtual_agent_pos = self.pos
        if action == 1: # UP
            virtual_agent_pos = [self.pos[0] - 1, self.pos[1]]
            # print("virtual_agent_pos: ", virtual_agent_pos)
            if self.posses_ball:
                virtual_ball_pos = virtual_agent_pos
        if action == 2: # Down
            virtual_agent_pos = [self.pos[0] + 1, self.pos[1]]
            # print("virtual_agent_pos: ", virtual_agent_pos)
            if self.posses_ball:
                virtual_ball_pos = virtual_agent_pos
        if action == 3: # Left
            virtual_agent_pos = [self.pos[0], self.pos[1] - 1]
            # print("virtual_agent_pos: ", virtual_agent_pos)
            if self.posses_ball:
                virtual_ball_pos = virtual_agent_pos
        if action == 4: # Right
            virtual_agent_pos = [self.pos[0], self.pos[1] + 1]
            # print("virtual_agent_pos: ", virtual_agent_pos)
            if self.posses_ball:
                virtual_ball_pos = virtual_agent_pos
        if self.team == "attack":
            agents_around, agents_pos_around = self.see_around(_map)
            if agents_pos_around:
                pass_agent_pos = random.choice(agents_pos_around)
                pass_agent_id = _map[pass_agent_pos[0]][pass_agent_pos[1]]
                if action == 5 and self.posses_ball: # Pass
                    # if the ball runs into boundary or defenders it will be stoped and one episode done
                    if self.log:
                        print("agent %d pass ball to agent %d" % (self.id, pass_agent_id))
                    virtual_ball_pos = ball.move(self.pos, pass_agent_pos, _map)
                    if ball.blocked:
                        # print("pass block 1")
                        block_agent_id = _map[virtual_ball_pos[0]][virtual_ball_pos[1]]
                        if agents[block_agent_id].team == "defend":
                            if self.log:
                                print("====pass blocked!====")
                            pass_blocked = True
                    else:
                        agents[pass_agent_id].posses_ball = True
            if action == 6 and self.posses_ball: # Shoot
                shoot_pos = self.can_shoot()
                if shoot_pos != None:
                    if self.log:
                        print("agent %d shoot" % self.id, " agent pos: ", self.pos)
                    # attack_gate_pos = self.gate_pos[1]
                    # print(attack_gate_pos)
                    # possible_shoot_pos = []
                    # for pos in attack_gate_pos:
                    #     row_start = pos[0] - 2 if pos[0] - 2 >= 0 else 0
                    #     row_end = pos[0] + 3 if pos[0] + 3 <= self.court_height else self.court_height
                    #     col_start = pos[1] - 2 if pos[1] - 2 >= 0 else 0
                    #     col_end = pos[1] + 3 if pos[1] + 3 <= self.court_width else self.court_width
                    #     for i in range(row_start, row_end):
                    #         for j in range(col_start, col_end):
                    #             possible_shoot_pos.append([i, j])

                    # attack_pos = random.choice(attack_gate_pos)
                    # there is a probability that shoot will miss
                    # possible_shoot_pos = []
                    # row_start = attack_pos[0] - 2 if attack_pos[0] - 2 >= 0 else 0
                    # row_end = attack_pos[0] + 3 if attack_pos[0] + 3 <= self.court_height else self.court_height
                    # col_start = attack_pos[1] - 2 if attack_pos[1] - 2 >= 0 else 0
                    # col_end = attack_pos[1] + 3 if attack_pos[1] + 3 <= self.court_width else self.court_width
                    # for i in range(row_start, row_end):
                    #     for j in range(col_start, col_end):
                    #         if i == attack_pos[0] and j == attack_pos[1]:
                    #             continue
                    #         possible_shoot_pos.append([i, j])
                    
                    # if random.random() > self.miss_prob:
                    #     shoot_pos = attack_pos
                    # else:
                    #     shoot_pos = random.choice(possible_shoot_pos)
                    # shoot_pos = attack_pos
                    if self.log:
                        print("shoot pos: ", shoot_pos)
                    virtual_ball_pos = ball.move(self.pos, shoot_pos, _map)
                    if ball.blocked:
                        # print("shoot block 1")
                        block_agent_id = _map[virtual_ball_pos[0]][virtual_ball_pos[1]]
                        if self.log:
                            print("block agent id: ", block_agent_id)
                        if agents[block_agent_id].team == "defend":
                            if self.log:
                                print("====shoot blocked!====")
                            shoot_blocked = True
                    if ball.check_ball_score(self.team, self.court_id, self.court_width, self.court_height, self.gate_width) == False:
                        # print("shoot not score.")
                        done = True
            if action == 5 or action == 6:
                virtual_agent_pos = self.pos
            # print("virtual ball pos: ", virtual_ball_pos)
        # print("virtual agent pos: ", virtual_agent_pos)
        # print("virtual ball pos: ", virtual_ball_pos)
        # print()

        return virtual_agent_pos, virtual_ball_pos, shoot_blocked, pass_blocked, done

    def after_step(self, action, _map, ball, agents):
        winner = None
        self.log = False
        stand_still_penalty = 0.0
        attack_reward, defend_reward, block_reward = 0.0, 0.0, 0.0
        reward_info = dict()
        # print("simulate in after step: ")
        virtual_agent_pos, virtual_ball_pos, shoot_blocked, pass_blocked, done = self.simulate_move(action, _map, ball, agents)
        self.log = True
        # check action = 0, we don't want the agent always stand still
        self.actions.append(action)
        action_records = {}
        for k, v, in itertools.groupby(self.actions):
            action_records[k] = list(v)
        if 0 in action_records.keys() and len(action_records[0]) >= 5:
            # if not move more than 3 steps give a penalty
            stand_still_penalty = -5.0
            self.actions.clear()
        if len(self.actions) > 1000:
            self.actions.clear() # avoid memory leak
        
        shoot_pos = self.can_shoot()

        if self.team == "attack":
            if self.court_id == "left" and action == 4:
                attack_reward += 1.0
            if self.court_id == "left" and action == 3:
                attack_reward += -1.0
            if self.court_id == "right" and action == 3:
                attack_reward += 1.0
            if self.court_id == "right" and action == 4:
                attack_reward += -1.0
            if action == 6 and shoot_pos != None:
                attack_reward += 7.0

        if self.team == "defend":
            if self.pos[0] - ball.pos[0] < 0 and action == 2:
                defend_reward += 1.0
            if  self.pos[0] - ball.pos[0] < 0 and action == 1:
                defend_reward += -1.0
            if self.pos[0] - ball.pos[0] > 0 and action == 1:
                defend_reward = 1.0
            if self.pos[0] - ball.pos[0] > 0 and action == 2:
                defend_reward += -1.0
            if self.pos[0] - ball.pos[0] == 0 and (action == 1 or action ==2):
                defend_reward += -1.0
            if self.pos[0] - ball.pos[0] == 0 and (action != 1 and action != 2):
                defend_reward += 1.0

            if self.pos[1] - ball.pos[1] < 0 and action == 4:
                defend_reward += 1.0
            if self.pos[1] - ball.pos[1] > 0 and action == 3:
                defend_reward += 1.0

            if pass_blocked or shoot_blocked:
                block_reward = 20.0
                winner = "defend"

        reward = attack_reward + defend_reward + block_reward + stand_still_penalty
        reward_info["attack"] = attack_reward
        reward_info["defend"] = defend_reward
        reward_info["block"] = block_reward
        reward_info["stand_still_penalty"] = stand_still_penalty
        
        return reward, reward_info, winner, virtual_agent_pos, virtual_ball_pos, done