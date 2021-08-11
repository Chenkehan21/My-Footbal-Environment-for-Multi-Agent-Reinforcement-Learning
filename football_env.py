import numpy as np
import random
from players import Players
from ball import Ball
import math
from collections import namedtuple
import time


'''

map:                             width: 23
            (0,0)------------------------------------------------> y
            |
            |
            |                      . agent_pos(x, y)
            |
            |--
height: 20  | | 
            | |gate range: 6
            | |
            | |
            |--
            |
            |
            |
            |
            x
 
x = agent_pos[0], y = agent_pos[1]

gate range:   [gate_center - 2, gate_center - 1, gate_center, gate_center + 1, gate_center + 2]

success rate: [0.3,             0.6,             0.9,         0.6,             0.3            ]

shoot range:  [agent_pos - 1, agent_pos, agent_pos + 1]

shoot distance: 1/4 * court_width

'''


class Football_Env:
    Observation = namedtuple('Observation', ['team', 'agent_id', 'obs'])
    Actions = namedtuple('Actions', ['team', 'agent_id', 'action'])
    Rewards = namedtuple('Rewards', ['team', 'agent_id', 'reward'])
    
    def __init__(self, 
    agents_left: list, agents_right: list, 
    max_episode_steps, move_reward_weight=1.0,
    court_width=23, court_height=20, gate_width = 6):
        self.agents_left = agents_left # list of agent_id in the left court
        self.agents_right = agents_right # list of agent_id in the right court
        self.max_episode_steps = max_episode_steps
        self.court_width = court_width
        self.court_height = court_height
        self.gate_width = gate_width
        self.move_reward_weight = move_reward_weight
        self.elapsed_steps = 0
        self.GOAL_REWARD = 10.0
        self.tackle_reward = 0.0
        self.score_defend_penalty = 0.0
        self.defender_block_reward = 0.0
        self.tackle_winner = None
        self.agents_conflict = False
        self.previous_map = None
        self._map = None
        self.n_agents = len(agents_left) + len(agents_right)
        self.agents = {}
        self.ball = Ball()
        self.Done = False
        self.attack_action_space_n = 7
        self.defend_action_space_n = 5
        self.blocked = False
        self.winner = []

    def reset(self):
        self.blocked = False
        self.Done = False
        self.elapsed_steps = 0
        self.tackle_reward = 0.0
        self.score_defend_penalty = 0.0
        self.defender_block_reward = 0.0
        self.tackle_winner = None
        self.winner = []

        # initialize map
        self.reset_map()

        # initializer team
        self.reset_agents_team()
        
        # intialize players postion
        for agent in self.agents.values():
            self.reset_agent_position(agent)

        # give ball to the attack team
        self.reset_ball_possesion()
        
        # update map 
        self.update_map()

        # for i in range(1, self.n_agents + 1):
        #     print("agent %d pos1: " % i, self.agents[i].pos, "|action: ", end='|')
        #     print("agent %d pos2: " % i, self.agents[i].pos, "|ball_pos", self.ball.pos, "|posses ball: ", self.agents[i].posses_ball)

        return self.get_obs()

    def reset_map(self):
        self.previous_map = np.zeros([self.court_height, self.court_width])
        self._map = np.zeros([self.court_height, self.court_width])

    def reset_agent_position(self, agent):
        left_court_start = 0
        steps = math.floor(self.court_width / 2)
        left_court_end = left_court_start + steps
        right_court_start = self.court_width - steps
        right_court_end = self.court_width
        gate_center = int(self.gate_width / 2)
        gate_range = list(range(int(self.court_height / 2) - int(self.gate_width / 2), 
                    int(self.court_height / 2) + int(self.gate_width / 2) + 1))

        # if agent.court_id == "left":
        #     y = random.choice(list(range(left_court_start, left_court_end))) # top left conner is (0, 0)
        #     x = random.choice(list(range(0, self.court_height)))
        #     if self._map[x][y] != 0:
        #         self.reset_agent_position(agent)
        #     else:
        #         agent.pos = [x, y]

        # if agent.court_id == "right":
        #     y = random.choice(list(range(right_court_start, right_court_end))) # top left conner is (0, 0)
        #     x = random.choice(list(range(0, self.court_height)))
        #     if self._map[x][y] != 0:
        #         self.reset_agent_position(agent)
        #     else:
        #         agent.pos = [x, y]



        if agent.court_id == "left":
            if agent.team == "attack":
                x = random.choice(list(range(0, self.court_height)))
                y = random.choice(list(range(left_court_start, int(left_court_end / 3))))
                if self._map[x][y] != 0:
                    self.reset_agent_position(agent)
                else:
                    agent.pos = [x, y]
            if agent.team == "defend":
                x = random.choice(gate_range)
                y = random.choice(list(range(left_court_start, int(left_court_end / 3))))
                if self._map[x][y] != 0:
                    self.reset_agent_position(agent)
                else:
                    agent.pos = [x, y]
            
        if agent.court_id == "right":
            if agent.team == "attack":
                x = random.choice(list(range(0, self.court_height)))
                y = random.choice(list(range(right_court_end - 3, right_court_end)))
                if self._map[x][y] != 0:
                    self.reset_agent_position(agent)
                else:
                    agent.pos = [x, y]
            if agent.team == "defend":
                x = random.choice(gate_range)
                y = random.choice(list(range(right_court_end - 5, right_court_end)))
                if self._map[x][y] != 0:
                    self.reset_agent_position(agent)
                else:
                    agent.pos = [x, y]

    def reset_agents_team(self):
        team_choices = ["attack", "defend"]
        # left_court_team = random.choice(team_choices)
        left_court_team = 'attack'
        team_choices.remove(left_court_team)
        right_court_team = team_choices[0]
        if left_court_team == "attack":
            self.attack_court = "left"
        else:
            self.attack_court = "right"
        
        for id in self.agents_left:
            self.agents[id] = Players(agent_id=id, court_id="left", team=left_court_team, 
            court_width=self.court_width, court_height=self.court_height, gate_width=self.gate_width,
            _map=self._map)
        for id in self.agents_right:
            self.agents[id] = Players(agent_id=id, court_id="right", team=left_court_team, 
            court_width=self.court_width, court_height=self.court_height, gate_width=self.gate_width,
            _map=self._map)

        for agent in self.agents.values():
            if agent.court_id == "left":
                agent.team = left_court_team
            if agent.court_id == "right":
                agent.team = right_court_team
    
    def reset_ball_possesion(self):
        for agent in self.agents.values():
            agent.posses_ball = False
            if agent.court_id == self.attack_court:
                self.ball.give_ball_possession(agent.pos)
                agent.posses_ball = True
                break

    def update_map(self):
        self.agents_conflict = False
        self.previous_map = self._map
        self._map = np.zeros([self.court_height, self.court_width])
        for agent in self.agents.values():
            if self._map[agent.pos[0]][agent.pos[1]] == 0:
                self._map[agent.pos[0]][agent.pos[1]] = agent.id
            else:
                print("^^^^^^^^^^agents contact!^^^^^^^^^^")
                self.agents_conflict = True
                self.winner.append('defend')
                self.tackle_reward = 5.0
                # agent_id = self._map[agent.pos[0]][agent.pos[1]]
                # if self.agents[agent_id].team != agent.team and (self.agents[agent_id].posses_ball or agent.posses_ball):
                #     self.tackle_reward = 5.0
                #     self.tackle_winner = 'defend'

    def get_obs(self):
        obs = []
        for i in range(1, self.n_agents + 1):
            _obs = self.Observation(self.agents[i].team, self.agents[i].id, self.agents[i]._get_obs(self._map))
            obs.append(_obs)
        
        return tuple(obs)

    def sample_actions(self):
        actions = []
        for i in range(1, self.n_agents + 1):
            _actions = self.Actions(self.agents[i].team, self.agents[i].id, self.agents[i].sample_action())
            actions.append(_actions)
        
        return tuple(actions)

    def agents_past_court(self, id):
        x = self.agents[id].pos[0]
        y = self.agents[id].pos[1]
        if x < 0 or x >= self.court_height or y < 0 or y >= self.court_width:
            return True
        return False

    def step(self, actions: tuple):
        self.winner = []
        self.Done = False
        self.blocked = False
        self.score_defend_penalty = 0.0
        done_rewards, move_rewards, goal_rewards = [], [], []
        dones = []
        infos = {"attack_team": [], "defend_team": [], "ball_pos": None, "winner": None}
        # print("self.n_agents: ", self.n_agents)
        for i in range(1, self.n_agents + 1):
            agent_info = {"id": None, "posses_ball": None, "reward_details": None, "move_details": None}
            agent_info["id"] = i
            agent_info["posses_ball"] = self.agents[i].posses_ball

            move_details = {"action": None, "last_position": None, "current_position": None}
            # print("actions: ", actions)
            move_details["action"] = actions[i - 1]
            move_details["last_position"] = self.agents[i].pos

            # if self.agents[i].team == 'defend':
            #     print("agent pos: ", self.agents[i].pos, " action: ", actions[i - 1])

            move_reward, rew_info, move_dones = self.get_move_rewards(self.agents[i], actions[i - 1])
            self.update_map()

            if self.agents[i].team == 'defend':
                if self.defender_block_reward != 0:
                    print("defender block reward: ", self.defender_block_reward)
                    # time.sleep(3)
                move_reward += self.defender_block_reward
                self.defender_block_reward = 0.0

            move_rewards.append(move_reward)
            dones += move_dones

            goal_reward = self.get_goal_rewards(self.agents[i], actions[i - 1])
            goal_rewards.append(goal_reward)
            if goal_reward != 0:
                dones.append(True)
            # print("agent %d pos1: " % i, self.agents[i].pos, "|action: ", actions[i - 1], end='|')
            agent_done, done_reward = self.get_done_rewards(self.agents[i], actions[i - 1])
            move_details["current_position"] = self.agents[i].pos
            # print("agent %d pos2: " % i, self.agents[i].pos, "|ball_pos", self.ball.pos, "|posses ball: ", self.agents[i].posses_ball)
            dones += agent_done
            done_rewards.append(done_reward)

            rew_info["done_reward"] = done_reward
            rew_info["goal_reward"] = goal_reward

            agent_info["reward_details"] = rew_info
            agent_info["move_details"] = move_details

            if self.agents[i].team == "attack":
                infos["attack_team"].append(agent_info)
            if self.agents[i].team == "defend":
                infos["defend_team"].append(agent_info)
            infos["ball_pos"] = self.ball.pos

        self.elapsed_steps += 1
        # print("elapsed steps: ", self.elapsed_steps)
        if self.elapsed_steps >= self.max_episode_steps:
            print("max episode steps elapsed!")
            dones.append(True)

        rewards = []

        # if self.GOAL_REWARD in goal_rewards or self.score_defend_penalty in goal_rewards and self.score_defend_penalty != 0:
        #     print("goal rewards: ", goal_rewards)
        #     time.sleep(1)

        for i in range(self.n_agents):
            rew = float(goal_rewards[i] + done_rewards[i] + self.move_reward_weight * move_rewards[i])
            reward = self.Rewards(team=self.agents[i + 1].team, agent_id=i + 1, reward=rew)
            rewards.append(reward)
        rewards = tuple(rewards)
        # print(dones)
        done = True in dones
        self.Done = done
        # if not done:
            # print("***update map***")
            # self.winner.append(self.tackle_winner)
        if done:
            print("\nself.winner: ", self.winner)
            if self.winner:
                infos['winner'] = self.winner[0]
            if len(self.winner) == 0 and self.elapsed_steps >= self.max_episode_steps:
                infos['winner'] = 'tie'
            for i in range(1, self.n_agents + 1):
                print("agent_%d pos: " % self.agents[i].id, self.agents[i].pos, "agent team: ", self.agents[i].team)

        obs = self.get_obs()
        # print("winner list: ", self.winner)

        return obs, rewards, done, infos

    def get_goal_rewards(self, agent, action):
        if action == 6 and agent.posses_ball:
            is_score = self.ball.check_ball_score(agent.team, agent.court_id, self.court_width, self.court_height, self.gate_width)
            if is_score and not self.blocked:
                print("==========shoot score!==========")
                # time.sleep(1)
                self.winner.append("attack")
                self.score_defend_penalty = -self.GOAL_REWARD
                return self.GOAL_REWARD
            else:
                # print("shoot not score.")
                self.winner.append("defend")
                return 0.0
        elif agent.team == 'defend' and self.score_defend_penalty != 0:
            return self.score_defend_penalty
        else:
            return 0.0

    def get_done_rewards(self, agent, action):
        dones = []
        done_reward = 0.0
        dones.append(self.agents_conflict)
        if self.agents_past_court(agent.id):
            print("agent %d past court!" % agent.id, ", agent team: ", agent.team)
            dones.append(True)
            done_reward -= 20.0
            if agent.team == "attack":
                self.winner.append("defend")
            if agent.team == "defend":
                self.winner.append("attack")
        
        # if self.ball.check_ball_pass_court(self._map):
        #     print("ball past court!")
        #     dones.append(True)
        #     done_reward -= 1.0
        #     if agent.team == "attack":
        #         self.winner.append("defend")
        #     if agent.team == "defend":
        #         self.winner.append("attack")

        # if self.ball.check_ball_score(agent.team, agent.court_id, self.court_width, self.court_height):
        #     print("hahahahahaha")
        #     dones.append(True)
        #     done_reward += 1.0
            

        return dones, done_reward

    def get_move_rewards(self, agent, action):
        dones = []
        move_reward, rew_info, winner, virtual_agent_pos, virtual_ball_pos, done, block = agent.after_step(action, self._map, self.ball, self.agents)
        self.agents[agent.id].pos = virtual_agent_pos
        self.ball.pos = virtual_ball_pos
        dones.append(done)
        # if winner != None:
        #     self.winner.append(winner)
        
        if block:
            self.blocked = True
            print("**********block**********")
            self.defender_block_reward = 100.0
            dones.append(True)
            self.winner.append('defend')
        rew_info['block'] = self.defender_block_reward

        return move_reward, rew_info, dones

    def get_all_pos(self):
        agents_pos = dict()
        for i in range(1, self.n_agents + 1):
            agents_pos[i] = self.agents[i].pos
        return agents_pos