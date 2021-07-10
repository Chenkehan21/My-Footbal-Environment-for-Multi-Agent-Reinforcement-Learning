import numpy as np
import random
from players import Players
from ball import Ball
import math

'''
(0,0)------------------> x
|
|
|     . pos(x, y)
|
|
|
y
'''

class Football_Env:
    def __init__(self, agents_left, agents_right, max_episode_steps, court_width, court_height):
        self.agents_left = agents_left
        self.agents_right = agents_right
        self.max_episode_steps = max_episode_steps
        self.court_width = court_width
        self.court_height = court_height
        self.elapsed_steps = 0
        self.GOAL_REWARD = 10
        self.agents_contact = False
        self.previous_map = None
        self._map = None
        self.n_agents = len(agents_left) + len(agents_right)
        self.agents = {}
        self.ball = Ball()

    def reset(self):
        self.previous_map = np.zeros([self.court_height, self.court_width])
        self._map = np.zeros([self.court_height, self.court_width])
        team_choices = ["attack", "defend"]
        left_court_team = random.choice(team_choices)
        team_choices.remove(left_court_team)
        right_court_team = team_choices[0]
        if left_court_team == "attack":
            self.attack_court = "left"
        else:
            self.attack_court = "right"
        
        id = 1
        for court in self.agents_left:
            self.agents[id] = Players(id, court_id=court, team=left_court_team, _map=self._map)
            id += 1
        id += 1
        for court in self.agents_right:
            self.agents[id] = Players(id, court_id=court, team=right_court_team, _map=self._map)

        for agent in self.agents.values:
            if agent.team == "attack":
                self.ball.give_ball_possession(agent.pos)
                agent.posses_ball = True
        self.update_map()

    def reset_agents_position(self, agent):
        left_court_start = 0
        left_court_end = math.floor(self.court_width / 2) + 1
        right_court_start = math.ceil(self.court_width / 2)
        right_court_end = self.court_width

        if self.court_id == "left":
            x = random.choice(list(range(left_court_start, left_court_end))) # top left conner is (0, 0)
            y = random.choice(list(range(0, self.court_height)))
            if self._map[x][y] != 0:
                self.reset_agents_position()
        if self.court_id == "right":
            x = random.choice(list(range(right_court_start, right_court_end))) # top left conner is (0, 0)
            y = random.choice(list(range(0, self.court_height)))
            if self._map[x][y] != 0:
                self.reset_agents_position()
        agent.pos = [x, y]

    def reset_agents(self):
        team_choices = ["attack", "defend"]
        left_court_team = random.choice(team_choices)
        team_choices.remove(left_court_team)
        right_court_team = team_choices[0]
        if left_court_team == "attack":
            attack_court = "left"
        else:
            attack_court = "right"
        for agent in self.agents.values():
            if agent.court_id == "left":
                agent.team = left_court_team
            if agent.court_id == "right":
                agent.team = right_court_team
            self.reset_agents_position(agent)
            self.update_map()
        for agent in self.agents.values():
            if agent.court_id == attack_court:
                self.ball.give_ball_possession(agent.pos)
                break
        
    def reset(self):
        self.previous_map = np.zeros([self.court_height, self.court_width])
        self._map = np.zeros([self.court_height, self.court_width])


    def update_map(self):
        self.previous_map = self._map
        self._map = np.zeros([self.court_height, self.court_width])
        for agent in self.agents.values():
            self._map[agent.pos[0]][agent.pos[1]] = agent.id

    def get_obs(self):
        obs = []
        for i in range(self.n_agents):
            obs.append(self.agents[i]._get_obs(self._map))
        return tuple(obs)

    def sample_actions(self):
        actions = []
        for i in range(self.n_agents):
            actions.append(self.agents[i].sample_action())
        return tuple(actions)

    def agents_past_court(self, id):
        x = self.agents[id].pos[1]
        y = self.agents[id].pos[0]
        if x < 0 or x >= self.court_width or y < 0 or y >= self.court_height:
            return True
        return False

    def agents_conflict(self):
        for i in range(self._map.shape[0]):
            for j in range(self._map.shape[1]):
                if self.previous_map[i][j] != 0 and self._map[i][j] != 0 and self.previous_map[i][j] != self._map[i][j]:
                    return True
        return False


    def goal_reward(self):
        self.elapsed_steps += 1
        agents_past_court = [self.agents_past_court(i) for i in range(self.n_agents)]
        agents_conflict = self.agents_conflict()
        ball_past_court = self.ball.check_ball_pass_court()
        score = self.ball.check_ball_score()


    def step(self, actions):
        for i in range(self.n_agents):
            move_reward, agent_done = self.agents[i].after_step(actions[i])

    def agent_pass_court(self, agent_pos):
        pass

    def _set_action_space(self):
        pass

    def _get_goal_rewards(self):
        pass

    def _get_done(self):
        pass



    def _get_obs(self):
        pass

    '''some rules
    '''
