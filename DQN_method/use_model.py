import torch

import os
import sys

from torch.utils.data.dataloader import DataLoader
sys.path.append(os.path.dirname(__file__) + os.sep + '../')

import random
import pickle

from football_env import Football_Env
from utils import *
from agents import Agents
from experience import Experience, CustomDataset


ATTACK_PATH = '/home/chenkehan/RESEARCH/codes/experiment4/DQN_method/res_attack5/football_0.710_59.780.dat'
DEFEND_PATH = '/home/chenkehan/RESEARCH/codes/experiment4/DQN_method/res_defend4/football_0.860_98.000.dat'
MODEL_PATH = 'haha'


def use_model():
    device = torch.device("cpu")
    env = Football_Env(agents_left=[1], agents_right=[2],
    max_episode_steps=500, move_reward_weight=1.0,
    court_width=23, court_height=20, gate_width=6)
    env.reset()
    
    my_experience = Experience(env, 'attack', device=device,
         use_trained_attack_net=True, use_trained_defend_net=True,
        defender_net_path=DEFEND_PATH, attacker_net_path=ATTACK_PATH, experience_size=40000)

    experience = my_experience.generate_experience()

    with open('/home/chenkehan/RESEARCH/codes/experiment4/DQN_method/test_set.pkl', 'wb') as f:
        pickle.dump(experience, f)



def test():
    device = torch.device("cpu")
    with open('/home/chenkehan/RESEARCH/codes/experiment4/DQN_method/models.pkl', 'rb') as f:
        models = pickle.load(f)

    with open('/home/chenkehan/RESEARCH/codes/experiment4/DQN_method/test_set.pkl', 'rb') as f:
        test_set = pickle.load(f)
    test_dataset = CustomDataset(test_set)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    actions  = []
    for model_id in range(12):
        root_path = '/home/chenkehan/RESEARCH/codes/experiment4/DQN_method/trained_models/'
        file_name = str(model_id) + '.pth'
        path = root_path + file_name

        net = torch.load(path)
        net.eval()    

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                values, action_batch = torch.max(outputs, dim=1)
                actions.append(action_batch)
        print(actions)
        print("\n\n")
        actions.append('\n')

    with open('/home/chenkehan/RESEARCH/codes/experiment4/DQN_method/actions.txt', 'w') as f:
        for action_batch in actions:
            for action in action_batch:
                if type(action) != str:
                    f.write(str(action.item()))
                else:
                    f.write(action)

if __name__ == "__main__":
    # use_model()
    test()