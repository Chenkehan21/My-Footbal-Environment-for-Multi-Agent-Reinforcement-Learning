import os
import sys
sys.path.append(os.path.dirname(__file__) + os.sep + '../')

import random
import pickle

import torch
import torch.nn as nn

from utils import*
from football_env import Football_Env


MODEL_NUM = 100


class Model(nn.Module):
    def __init__(self, linear_num, hidden_size, act_layer, obs_size, act_size):
        super(Model, self).__init__()
        self.logstd = nn.Parameter(torch.zeros(act_size))
        self.mu = self._make_layer(linear_num, hidden_size, act_layer, obs_size, act_size)

    # linear_num is random, hidden_size is random, act_layer is a list we use random choice
    def _make_layer(self, linear_num, hidden_size, act_layer, obs_size, act_size):
        layer_list = nn.ModuleList()

        first_layer = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            random.choice(act_layer),
        )
        layer_list.append(first_layer)
        for _ in range(linear_num):
            hidden_layer = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                random.choice(act_layer),
            )
            layer_list.append(hidden_layer)
        linear_layer = nn.Sequential(
            nn.Linear(hidden_size, act_size),
            random.choice(act_layer),
        )
        layer_list.append(linear_layer)
        last_layer = nn.LogSoftmax(dim=1)
        layer_list.append(last_layer)

        return layer_list

    def forward(self, x):
        for layer in self.mu:
            x = layer(x)
        return x


def build_models(env, train_team='attack', model_num=MODEL_NUM):
    if train_team == "attack":
        output_shape = env.attack_action_space_n
    if train_team == "defend":
        output_shape = env.defend_action_space_n

    # env.reset()
    # state = env._map
    # input_shape = state.size

    state = env.reset()
    state = handle_obs(state, 'attack')
    input_shape = len(state)

    models = []
    for _ in range(model_num):
        linear_num = random.randrange(1, 10)
        hidden_size = random.randrange(16, 512)
        act_layer = [nn.ReLU(), nn.Tanh(), nn.Sigmoid(), nn.LeakyReLU(), nn.ReLU6(), nn.RReLU()]
        model = Model(linear_num, hidden_size, act_layer, input_shape, output_shape)
        models.append(model)

    return models


if __name__ == '__main__':
    env = Football_Env(agents_left=[1], agents_right=[2],
    max_episode_steps=500, move_reward_weight=1.0,
    court_width=23, court_height=20, gate_width=6)

    env.reset()
    models = build_models(env)
    with open('./models.pkl', 'wb') as f:
        pickle.dump(models, f)