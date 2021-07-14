from football_env import Football_Env


if __name__ == "__main__":
    env = Football_Env(agents_left=[1, 2], agents_right=[3, 4],
    max_episode_steps=500, move_reward_weight=1.0,
    court_width=24, court_height=18, gate_width=6)
    env.reset()
    