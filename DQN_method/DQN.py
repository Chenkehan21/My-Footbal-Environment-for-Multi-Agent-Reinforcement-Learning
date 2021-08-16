import torch
from torch._C import dtype
import torch.nn as nn
from torch.nn.modules import loss
import torch.optim as optim

import numpy as np
import random
import collections
import argparse
import time
import os
import sys
sys.path.append(os.path.dirname(__file__) + os.sep + '../')

from football_env import Football_Env
from DQN_method.DQN_model import DQN
from utils import *

from tensorboardX import SummaryWriter


# hyperparameters
DEFAULT_ENV_NAME = "football"
MEAN_REWARD_BOUND = 50 # pong game end when an agent gets 21 points

GAMMA = 0.99 # used for Bellman approximation
BATCH_SIZE = 256 # the size sampled from replay buffer
REPLAY_SIZE = 10000 # the maximum capicity of replay buffer
REPLAY_START_SIZE = 10000 # the counts of frames we wait before start training
LEARNING_RATE = 1e-4 # learning rate used in Adam optimizer
SYNC_TARGET_NET = 1000 # how frequently we sync target net

EPSILON_DECAY_LAST_FRAMES = 40000 # during the first 150,000 frames, epsilon is linearly decayed to 0.01
EPSILON_START = 1.0
EPSILON_END = 0.01

SAVE_PATH_ATTACK = './DQN_method/res_attack6'
SAVE_PATH_DEFEND = './DQN_method/res_defend4'


Experience = collections.namedtuple("Experience", field_names=["state", "action", "reward", "done", "next_state"])


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size=BATCH_SIZE):
        index_mask = np.random.choice(len(self.buffer), size=batch_size, replace=False) # replace=False means no repetition
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in index_mask]) # nice trick!

        return np.array(states), np.array(actions), \
               np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), \
               np.array(next_states)


class Agent:
    def __init__(self, env, exp_buffer, train_team):
        self.env = env
        self.exp_buffer = exp_buffer
        self.train_team = train_team
        if train_team == 'attack':
            self.AI_team = 'defend'
        else:
            self.AI_team = 'attack'
        self._reset()

    def _reset(self):
        state = self.env.reset()
        self.state = handle_obs(state, self.train_team)
        if self.train_team == 'attack':
            self.AI_state = handle_obs(state, 'defend')
        if self.train_team == 'defend':
            self.AI_state = handle_obs(state, 'attack')
        self.total_rewards = 0.0

    def check_action(self, action, team):
        for agent in self.env.agents.values():
            if team == "attack" and agent.team == team:
                actions = list(range(7))
                actions.remove(5)
                if agent.pos[0] == 0:
                    actions.remove(1)
                if agent.pos[0] == self.env.court_height - 1:
                    actions.remove(2)
                if agent.pos[1] == 0:
                    actions.remove(3)
                if agent.pos[1] == self.env.court_width - 1:
                    actions.remove(4)

                # shoot_pos = agent.can_shoot()
                # if shoot_pos:
                #     shoot_success_rate = agent.success_rate[shoot_pos[0]]
                #     if (agent.posses_ball and shoot_pos and random.random() < shoot_success_rate) != True:
                #         actions.remove(6)
                # else:
                #     actions.remove(6)
                
                if action in actions:
                    return True

            if team == "defend" and agent.team == team:
                actions = list(range(5))
                if agent.pos[0] == 0:
                    actions.remove(1)
                if agent.pos[0] == self.env.court_height - 1:
                    actions.remove(2)
                if agent.pos[1] == 0:
                    actions.remove(3)
                if agent.pos[1] == self.env.court_width - 1:
                    actions.remove(4)
                
                if action in actions:
                    return True
        return False

    @torch.no_grad()
    def play_one_step(self, net, epsilon, device, trained_defend_net, trained_attack_net):
        # print("epsilon: ", epsilon)
        done_reward = None
        win_times = 0
        # choose action using epsilon greedy policy
        if np.random.rand() < epsilon:
            total_actions = self.env.sample_actions()
            actions = []
            for action in total_actions:
                if action.team == self.train_team:
                    trainer_action = action.action
                actions.append(action.action)
            # print("actions here: ", actions)
        else:
            state_a = np.array(self.state, copy=False) # add an dimension for BATCH_SIZE!
            state_v = torch.tensor(state_a, dtype=torch.float).to(device)
            AI_state_a = np.array(self.AI_state, copy=False)
            AI_state_v = torch.tensor(AI_state_a, dtype=torch.float).to(device)

            # print(state_v, '  ', state_v.shape)
            q_values = net(state_v) # since we put in state_v, batch_size=1 so q_values's size is [1, action_space.n]
            if len(q_values) > 0:
                sorted_q_values, index = q_values.sort(descending=True)
                trainer_actions = index.tolist()
                while self.check_action(trainer_actions[0], self.train_team) == False:
                    trainer_actions.pop(0)
                trainer_action = trainer_actions[0]

            AI_action, AI_q_values = None, []
            if self.train_team == 'attack' and trained_defend_net:
                AI_q_values = trained_defend_net(AI_state_v)
            
            if self.train_team == 'defend' and trained_attack_net:
                AI_q_values = trained_attack_net(AI_state_v)

            if len(AI_q_values) > 0:
                sorted_q_values, index = AI_q_values.sort(descending=True)
                AI_actions = index.tolist()
                while self.check_action(AI_actions[0], self.AI_team) == False:
                    # print("AI actions: ", AI_actions)
                    AI_actions.pop(0)
                AI_action = AI_actions[0]
            # print("AI action: ", AI_action, "trainer action: ", trainer_action)
            total_actions = self.env.sample_actions()
            actions = []
            for action in total_actions:
                if action.team == self.train_team:
                    actions.append(trainer_action)
                elif AI_action:
                    # print("AI action: ", AI_action)
                    actions.append(AI_action)
                else:
                    actions.append(action.action)
            # print("actions here2: ", actions)
        
        # step action
        # print("actions: ", actions)
        next_state, rewards, done, info = self.env.step(actions)
        for rew in rewards:
            if rew.team == self.train_team:
                reward = rew.reward
        trainer_next_state = handle_obs(next_state, self.train_team)
        exp = Experience(state=self.state, action=trainer_action, reward=reward, done=done, next_state=trainer_next_state)
        self.exp_buffer.append(exp)
        self.total_rewards += reward
        if done:
            print("winner: ", info['winner'])
            if info['winner'] == self.train_team:
                win_times += 1
            done_reward = self.total_rewards
            self._reset()
        self.state = trainer_next_state
        if self.train_team == 'attack':
            self.AI_state = handle_obs(next_state, 'defend')
        if self.train_team == 'defend':
            self.AI_state = handle_obs(next_state, 'attack')

        return done_reward, win_times


def calc_loss(loss_function, batch, net, target_net, device="cpu"):
    states, actions, rewards, dones, next_states = batch
    states_v = torch.tensor(np.array(states, copy=False), dtype=torch.float).to(device) # copy=False means if states changes states_v will change too
    actions_v = torch.tensor(np.array(actions, copy=False), dtype=torch.int64).to(device) # use copy can save memory
    rewards_v = torch.tensor(np.array(rewards, copy=False), dtype=torch.float).to(device)
    done_mask = torch.BoolTensor(dones).to(device)
    next_states_v = torch.tensor(np.array(next_states, copy=False), dtype=torch.float).to(device)

    # print("actions_v: ", actions_v)
    q_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

    with torch.no_grad():
        max_next_q_values = target_net(next_states_v).max(1)[0] # maxmize along axis 1, return values and index. we need values only
        
        max_next_q_values[done_mask] = 0.0

        max_next_q_values = max_next_q_values.detach()

    expected_value = rewards_v + GAMMA * max_next_q_values
    # print(q_values, expected_value)
    return loss_function(q_values, expected_value)


def train(net, target_net, buffer, agent, optimizer, loss_function, 
         device, save_path, trained_defend_net=None, trained_attack_net=None):
    print("start training")
    writer = SummaryWriter(comment='-'+DEFAULT_ENV_NAME)
    frame_idx = 0
    ts_frame = 0
    best_reward = -1e7
    best_win_rate = -1.0
    total_rewards = []
    total_win_times = []
    ts = time.time()
    while True:
        frame_idx += 1
        epsilon = max(EPSILON_END, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAMES)
        reward, win_times = agent.play_one_step(net, epsilon, device, trained_defend_net, trained_attack_net) 
        # when play steps buffer will be populated
        if reward is not None:
            total_rewards.append(reward)
            total_win_times.append(win_times)
            # print('total_win_times: ', total_win_times)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            mean_reward = np.mean(total_rewards[-100:])
            win_rate = np.mean(total_win_times[-100:])
            # print("total_reward: ", total_rewards)

            print("total steps: %d| %d games done| mean_reward100: %.3f| win_rate: %.3f| eps: %.3f| speed: %.3f f/s" %\
                (frame_idx, len(total_rewards), mean_reward, win_rate, epsilon, speed))
            # time.sleep(2)

            if len(total_rewards) > 100:
                writer.add_scalar("dqn_"+DEFAULT_ENV_NAME+"_mean_reward", mean_reward, len(total_rewards))
                writer.add_scalar("dqn_"+DEFAULT_ENV_NAME+"_win_rate", win_rate, len(total_win_times))
            
            if best_reward < mean_reward and len(total_rewards) > 100:
                torch.save(net.state_dict(), save_path+"/"+DEFAULT_ENV_NAME+"_%.3f_%.3f.dat" % (mean_reward, win_rate))
                print("best reward update: %.3f -> %.3f, win rate: %.3f" % (best_reward, mean_reward, win_rate))
                best_reward = mean_reward
            
            if best_win_rate < win_rate and len(total_win_times) > 100 and mean_reward > 0:
                torch.save(net.state_dict(), save_path+"/"+DEFAULT_ENV_NAME+"_%.3f_%.3f.dat" % (win_rate, mean_reward))
                print("best win rate update: %.3f -> %.3f, mean reward: %.3f" % (best_win_rate, win_rate, mean_reward))
                best_win_rate = win_rate

            if best_win_rate == win_rate and best_reward < mean_reward and len(total_rewards) > 100:
                torch.save(net.state_dict(), save_path+"/"+DEFAULT_ENV_NAME+"_%.3f_%.3f.dat" % (win_rate, mean_reward))
                print("keep best win rate: %.3f, best reward update: %.3f -> %.3f" % (best_win_rate, best_reward, mean_reward))
                best_reward = mean_reward

            # if mean_reward > MEAN_REWARD_BOUND:
            #     print("solved in %d frames" % frame_idx)
            #     break

        if len(buffer) < BATCH_SIZE:
            continue
        
        # Don't forget sync target net otherwise it won't converge!
        if frame_idx % SYNC_TARGET_NET == 0:
            target_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()
        batch = buffer.sample()
        loss = calc_loss(loss_function, batch, net, target_net, device)
        writer.add_scalar("dqn_"+DEFAULT_ENV_NAME+"_train_loss", loss, frame_idx)
        # print(loss)
        loss.backward()
        optimizer.step()
    
        # writer.close()
        

def main(save_path, train_team, use_trained_defend_net=False, use_trained_attack_net=False):
    torch.manual_seed(1234)
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda computation")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME, help="default env name")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    os.makedirs(save_path, exist_ok=True)

    env = Football_Env(agents_left=[1], agents_right=[2],
    max_episode_steps=5000, move_reward_weight=1.0,
    court_width=23, court_height=20, gate_width=6)

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer, train_team)

    if agent.train_team == "attack":
        action_space = agent.env.attack_action_space_n
    if agent.train_team == "defend":
        action_space = agent.env.defend_action_space_n

    input_shape = len(agent.state)
    output_shape = action_space 
    net = DQN(input_shape, output_shape).to(device)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE) # only need one optimizer
    loss_function = nn.MSELoss()
    
    target_net = DQN(input_shape, output_shape).to(device)
    print(net)

    if use_trained_defend_net:
        output_shape = agent.env.defend_action_space_n
        defender_net_path = '/home/chenkehan/RESEARCH/codes/experiment4/DQN_method/res_defend4/football_0.860_98.000.dat'
        trained_defend_net = DQN(input_shape, output_shape).to(device)
        trained_defend_net.eval()
        if defender_net_path:
            trained_defend_net.load_state_dict(torch.load(defender_net_path))
            print(trained_defend_net)

    if use_trained_attack_net:
        output_shape = agent.env.attack_action_space_n
        attacker_net_path = '/home/chenkehan/RESEARCH/codes/experiment4/DQN_method/res_attack5/football_0.710_59.780.dat'
        trained_attack_net = DQN(input_shape, output_shape).to(device)
        trained_attack_net.eval()
        if attacker_net_path:
            trained_attack_net.load_state_dict(torch.load(attacker_net_path))

    if not use_trained_attack_net and not use_trained_defend_net:
        train(net, target_net, buffer, agent, optimizer, loss_function,
        device, save_path, trained_defend_net=None, trained_attack_net=None)
    if use_trained_attack_net:
        train(net, target_net, buffer, agent, optimizer, loss_function,
        device, save_path, trained_defend_net=None, trained_attack_net=trained_attack_net)
    if use_trained_defend_net:
        train(net, target_net, buffer, agent, optimizer, loss_function,
        device, save_path, trained_defend_net=trained_defend_net, trained_attack_net=None)

if __name__ == "__main__":
    main(save_path=SAVE_PATH_ATTACK, train_team='attack',
         use_trained_attack_net=False, use_trained_defend_net=True)