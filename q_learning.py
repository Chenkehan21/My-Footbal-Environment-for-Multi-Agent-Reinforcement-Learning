from numpy.core.fromnumeric import mean
from football_env import Football_Env
import collections
from utils import *
import time
import pickle


GAMMA = 0.9
ALPHA = 0.2
TEST_EPISODES = 100

# train attack one on one
class Agent:
    """q-learning just need q_table, we don't bother to calculate dynamic fuction.  
    """
    def __init__(self, train_team="defend"):
        self.env = Football_Env(agents_left=[1], agents_right=[2],
        max_episode_steps=50000, move_reward_weight=1.0,
        court_width=23, court_height=20, gate_width=6)
        self.train_team = train_team
        state = self.env.reset()
        self.state = handle_obs(state, self.train_team)
        self.q_value_table = collections.defaultdict(float)
        
    
    """to get s, a, r, next_s
    """
    def sample_env(self):
        total_actions = self.env.sample_actions()
        train_action = handle_actions(total_actions, self.train_team)
        actions = []
        for action in total_actions:
            actions.append(action.action)
        old_state = self.state
        new_state, reward, done, _ = self.env.step(actions)
        new_state = handle_obs(new_state, self.train_team)
        if done:
            state = self.env.reset()
            self.state = handle_obs(state, self.train_team)
        else:
            self.state = new_state
        
        return old_state, train_action, reward, new_state

    """we don't need to traverse all states, we just check all possible actions and choose the best one
    """
    def get_best_v_and_best_a(self, state):
        best_value, best_action = -100, -1
        if self.train_team == "attack":
            action_space = self.env.attack_action_space_n
        if self.train_team == "defend":
            action_space = self.env.defend_action_space_n
        for action in range(action_space):
            q_value = self.q_value_table[(state.tobytes(), action)]
            if q_value > best_value:
                best_value = q_value
                best_action = action
        return best_value, best_action

    def update_q_value_table(self, s, a, r, s_next):
        old_q_value = self.q_value_table[(s.tobytes(), a[0])]
        new_q_value, _ = self.get_best_v_and_best_a(s_next)
        # print("===a: ", a)
        # print("old_q_value: ", old_q_value)
        # print("new_q_value: ", new_q_value)
        # print("r: ", r)
        for rew in r:
            if rew.team == self.train_team:
                reward = rew.reward
        self.q_value_table[(s.tobytes(), a[0])] = (1 - ALPHA) * old_q_value + ALPHA * (reward + GAMMA * new_q_value)
        # self.q_value_table[(s.tobytes(), a)] = 2

    def play_episode(self, env):
        print(self.state)
        total_reward = 0.0
        win_times = 0
        total_steps = 0
        state = env.reset()
        state = handle_obs(state, self.train_team)
        while True:
            _, trainer_action = self.get_best_v_and_best_a(state)
            total_actions = self.env.sample_actions()
            # print("trainer action: ", trainer_action)
            # print("total actions: ", total_actions)
            print("trainer action: ", trainer_action)
            actions = []
            for action in total_actions:
                if action.team == self.train_team:
                    actions.append(trainer_action)
                else:
                    actions.append(action.action)
            # print("actions: ", actions)
            new_state, reward, done, info = env.step(actions)
            total_steps += 1
            for rew in reward:
                if rew.team == self.train_team:
                    total_reward += rew.reward
            if done:
                # print("====done====", "    winner: ", info['winner'], '\n')
                if info['winner'] == self.train_team:
                    win_times += 1
                state = env.reset()
                state = handle_obs(state, self.train_team)
                break
            state = handle_obs(new_state, self.train_team)
        
        return total_reward, win_times, total_steps

def main():
    env = Football_Env(agents_left=[1], agents_right=[2],
    max_episode_steps=50000, move_reward_weight=1.0,
    court_width=23, court_height=20, gate_width=6)
    iter_n = 0
    best_win_rate = -1.0
    best_reward = -1e8
    agent = Agent()
    for _ in range(100):
        # print(agent.q_value_table)
        iter_n += 1
        total_reward = 0.0
        total_win = 0
        total_steps = 0
        s, a, r, next_s = agent.sample_env()
        # print("a: ", a)
        agent.update_q_value_table(s, a, r, next_s)
        for i in range(TEST_EPISODES):
            rew, win_n, steps = agent.play_episode(env)
            total_reward += rew
            total_win += win_n
            total_steps += steps
        mean_win_rate = total_win / TEST_EPISODES
        mean_steps = total_steps / TEST_EPISODES
        mean_reward = total_reward / TEST_EPISODES
        # if mean_reward > 0.8:
        #     print("solved in %d iterations, final reward %.3f" % (iter_n, mean_reward))
        #     break
        # print("mean steps:", mean_steps, "    defend total reward:", total_reward, 
        #      "    total win:", total_win, "    table size: ", len(agent.q_value_table))
        if mean_win_rate > best_win_rate:
            # print("best win rate update: %.3f -> %.3f" % (best_win_rate, mean_win_rate))
            best_win_rate = mean_win_rate
            # with open('./q_learning_res/q_learning_%.3f_%.3f.pkl' % (best_reward, best_win_rate), 'wb') as f:
            #     pickle.dump(agent, f)
            # time.sleep(1)
        if mean_reward > best_reward:
            print("n: ", iter_n, "best reward update: %.3f -> %.3f" % (best_reward, mean_reward), "defend win rate: ", best_win_rate)
            best_reward = mean_reward
            with open('./q_learning_res/q_learning_%.3f_%.3f.pkl' % (best_reward, best_win_rate), 'wb') as f:
                pickle.dump(agent, f)
            time.sleep(2)
   

if __name__ == "__main__":
    for i in range(100):
        main()       