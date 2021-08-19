import numpy as np

def handle_obs(all_obs, train_team):
    res = []
    for obs_info in all_obs:
        obs = obs_info.obs
        if obs_info.team == train_team:
            res.append(obs)
    # if len(res) > 1:
    # res = np.concatenate(res)
    
    return res

def handle_actions(all_actions, train_team):
    res = []
    for action_info in all_actions:
        action = action_info.action
        if action_info.team == train_team:
            res.append(action)
    
    return res