from football_env import Football_Env
import time

env = Football_Env(agents_left=[1], agents_right=[2],
    max_episode_steps=50000, move_reward_weight=1.0,
    court_width=23, court_height=20, gate_width=6)

n = 1
while True:
    print(n)
    attack_total_reward = 0.0
    defend_total_reward = 0.0
    total_steps = 0
    attack_total_win = 0
    defend_total_win = 0
    for i in range(100):
        state = env.reset()
        while True:
            all_actions = env.sample_actions()
            actions = []
            for action in all_actions:
                actions.append(action.action)
            obs, rewards, done, info = env.step(actions)
            total_steps += 1
            # print("total steps: ", total_steps)
            for reward in rewards:
                if reward.team == "attack":
                    attack_total_reward += reward.reward
                if reward.team == "defend":
                    defend_total_reward += reward.reward
            if done:
                # print("****done****")
                # print("attack total reward: ", attack_total_reward)
                # print("defend total reward: ", defend_total_reward)
                # print()
                if info['winner'] == 'attack':
                    attack_total_win += 1
                if info['winner'] == 'defend':
                    defend_total_win += 1
                break
    mean_attack_reward = attack_total_reward / 100
    mean_defend_reward = defend_total_reward / 100
    attack_win_rate = attack_total_win / 100
    defend_win_rate = defend_total_win / 100
    mean_steps = total_steps / 100
    print("mean attack reward: ", mean_attack_reward)
    print("mean defend reward: ", mean_defend_reward)
    print("attack win rate: ", attack_win_rate)
    print("defend win rate: ", defend_win_rate)
    n += 1
    time.sleep(3)
    # if mean_defend_reward > 26:
    #     break