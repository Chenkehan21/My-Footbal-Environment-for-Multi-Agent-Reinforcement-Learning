import numpy as np
import random
from players import Players
from ball import Ball
import math
from collections import namedtuple


'''
map:
(0,0)------------------> y
|
|
|     . pos(x, y)
|
|
|
x

x = pos[0], y = pos[1]
'''

Observation = namedtuple('Observation', ['team', 'agent_id', 'obs'])
Actions = namedtuple('Actions', ['team', 'agent_id', 'actions'])
Rewards = namedtuple('Rewards', ['team', 'agent_id', 'rewards'])


class Football_Env:
    def __init__(self, 
    agents_left: list, agents_right: list, 
    max_episode_steps, move_reward_weight=1.0,
    court_width=24, court_height=18, gate_width = 6):
        self.agents_left = agents_left # list of agent_id in the left court
        self.agents_right = agents_right # list of agent_id in the right court
        self.max_episode_steps = max_episode_steps
        self.court_width = court_width
        self.court_height = court_height
        self.gate_width = gate_width
        self.move_reward_weight = move_reward_weight
        self.elapsed_steps = 0
        self.GOAL_REWARD = 10.0
        self.agents_conflict = False
        self.previous_map = None
        self._map = None
        self.n_agents = len(agents_left) + len(agents_right)
        self.agents = {}
        self.ball = Ball()

    def reset(self):
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

    def reset_map(self):
        self.previous_map = np.zeros([self.court_height, self.court_width])
        self._map = np.zeros([self.court_height, self.court_width])

    def reset_agent_position(self, agent):
        left_court_start = 0
        steps = math.floor(self.court_width / 2)
        left_court_end = left_court_start + steps
        right_court_start = self.court_width - steps
        right_court_end = self.court_width

        if agent.court_id == "left":
            y = random.choice(list(range(left_court_start, left_court_end))) # top left conner is (0, 0)
            x = random.choice(list(range(0, self.court_height)))
            if self._map[x][y] != 0:
                self.reset_agent_position(agent)
            else:
                agent.pos = [x, y]
        if agent.court_id == "right":
            y = random.choice(list(range(right_court_start, right_court_end))) # top left conner is (0, 0)
            x = random.choice(list(range(0, self.court_height)))
            if self._map[x][y] != 0:
                self.reset_agent_position(agent)
            else:
                agent.pos = [x, y]

    def reset_agents_team(self):
        team_choices = ["attack", "defend"]
        left_court_team = random.choice(team_choices)
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
                print("agents contact!")
                self.agents_conflict = True

    def get_obs(self):
        obs = []
        for i in range(self.n_agents):
            _obs = Observation(self.agents[i].team, self.agents[i].id, self.agents[i]._get_obs(self._map))
            obs.append(_obs)
        
        return tuple(obs)

    def sample_actions(self):
        actions = []
        for i in range(self.n_agents):
            _actions = Actions(self.agents[i].team, self.agents[i].id, self.agents[i]._sample_action())
            actions.append(_actions)
        
        return tuple(actions)

    def agents_past_court(self, id):
        x = self.agents[id].pos[0]
        y = self.agents[id].pos[1]
        if x < 0 or x >= self.court_width or y < 0 or y >= self.court_height:
            return True
        return False

    def step(self, actions: list):
        self.elapsed_steps += 1
        done_rewards, move_rewards, goal_rewards = [], [], []
        dones = []
        infos = []
        for i in range(self.n_agents):
            agent_done, done_reward = self.get_done_rewards(self.agents[i], actions[i])
            dones += agent_done
            done_rewards.append(done_reward)
            move_reward, rew_info = self.get_move_rewards(self.agents[i], actions[i])
            move_rewards.append(move_reward)
            goal_reward = self.get_goal_rewards(actions[i])
            goal_rewards.append(goal_reward)
            rew_info["done_reward"] = done_reward
            rew_info["goal_reward"] = goal_reward
            infos.append(rew_info)

        rewards = []
        for i in range(self.n_agents):
            rew = float(goal_rewards[i] + done_rewards[i] + self.move_reward_weight * move_rewards[i])
            reward = Rewards(team=self.agents[i].team, agent_id=i, rewards=rew)
            rewards.append(reward)
        rewards = tuple(rewards)
        done = True in dones
        obs = self.get_obs()
        infos = tuple(infos)

        return obs, rewards, done, infos

    def get_goal_rewards(self, action):
        if action == 6:
            is_score = self.ball.check_ball_score()
            if is_score:
                return self.GOAL_REWARD
            else:
                return 0.0
        else:
            return 0.0

    def get_done_rewards(self, agent, action):
        dones = []
        done_reward = 0.0
        agent.pos, self.ball.pos = agent.simulate_move(action, self._map, self.ball)
        self.update_map()
        dones.append(self.agents_conflict)
        if self.agents_past_court(agent.id):
            dones.append(True)
            done_reward -= 1.0
        if self.ball.check_ball_pass_court():
            dones.append(True)
            done_reward -= 1.0
        if self.ball.check_ball_score(agent.team, agent.court_id, self.court_width, self.court_height):
            dones.append(True)
            done_reward += 1.0

        return dones, done_reward

    def get_move_rewards(self, agent, action):
        move_reward, rew_info = agent.after_step(action)
        
        return move_reward, rew_info

    def get_all_pos(self):
        agents_pos = dict()
        for i in range(1, self.n_agents + 1):
            agents_pos[i] = self.agents[i].pos
            print("agent %d pos: " % i, self.agents[i].pos)
        return agents_pos