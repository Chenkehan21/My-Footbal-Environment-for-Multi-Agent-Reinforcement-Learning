import os
import sys
sys.path.append(os.path.dirname(__file__) + os.sep + '../')

from football_env import Football_Env

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import random
import pickle
import pandas as pd
from scipy.spatial import distance

from tensorboardX import SummaryWriter
import argparse

from experience import Experience, CustomDataset


TRAIN_SET_PERCENTAGE = 0.7
BATCH_SIZE = 64
EPOCH = 200
LEARNING_RATE = 0.001
REPORT_EVERY_BATCH = 500
HISTORY_NUM = 60000
MODEL_NUM = 100
DIS_THRESHOLD = 0.985
TEST_THRESHOLD = 0.985

# train
def train(net, train_loader, criterion, optimizer, device, model_id):
    print("using device: {}".format(device))
    print("start training")
    writer = SummaryWriter(comment="%d" % model_id)
    iter_batch = 0

    for epoch in range(EPOCH):
        net.train() # remember to set train mode in every iteration! https://zhuanlan.zhihu.com/p/302409233
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            # print("outputs: ", outputs)
            # print("labels: ", labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() # .item() can get the value of one element tensors
            iter_batch += 1
            if i % REPORT_EVERY_BATCH == (REPORT_EVERY_BATCH - 1): # otherwise the first loss will be very small it's not true
                writer.add_scalar("loss", running_loss / REPORT_EVERY_BATCH, iter_batch)
                print("model %d|  epoch:%2d|  batch: %5d|  loss: %.3f" % (model_id, epoch + 1, i + 1, running_loss / REPORT_EVERY_BATCH))
                running_loss = 0.0
    writer.close()
    path = "./model_data/" + str(model_id) + ".pth"
    torch.save(net, path)
    print("model%d Done!" % model_id)
    return path


def test(net, test_loader, device, path, model_id, distance_threshold):
    print("start testing")
    # net.eval()
    # net.load_state_dict(torch.load(path))
    net = torch.load(path)
    net.eval()    
    total, correct = 0, 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = net(inputs)
            outputs = outputs.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            for i in range(len(outputs)):
                cos_dis = 1 - distance.cosine(outputs[i], labels[i])
                # print("cos_dis: ", cos_dis)
                if cos_dis >= distance_threshold:
                    correct += 1
            total += labels.shape[0]

    accuracy = correct / total
    print("model %d Accuracy: %.3f %%" % (model_id, 100 * accuracy))
    return accuracy


def main(to_train=True, to_test=False, test_trained_models=False, history_num=HISTORY_NUM, model_num=MODEL_NUM, dis_threshold=DIS_THRESHOLD):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda", default=False, action="store_true",
        help="Enable cuda computation"
    )
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    torch.manual_seed(1)

    env = Football_Env(agents_left=[1], agents_right=[2],
    max_episode_steps=500, move_reward_weight=1.0,
    court_width=23, court_height=20, gate_width=6)

    env.reset()
    # experience = ExperienceSource(env, history_num, model_num)
    # histories, models = experience.generate_history()
    # with open('./histories.pickle', 'wb') as f:
    #     pickle.dump(histories, f)

    # with open('./models.pickle', 'wb') as f:
    #     pickle.dump(models, f)

    with open('./experiences.pickle', 'rb') as f:
        histories = pickle.load(f)
    
    with open('./models.pickle', 'rb') as f:
        models = pickle.load(f)

    # prepare dataset
    states0, actions0 = [], []
    for history in histories:
        state0, action0 = history
        states0.append(state0)
        actions0.append(action0)

    # split train_set and test_set
    dataset_size = len(states0)

    train_data = histories[: int(TRAIN_SET_PERCENTAGE) * dataset_size]
    train_dataset = CustomDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_data = histories[int(TRAIN_SET_PERCENTAGE * dataset_size): ]
    test_dataset = CustomDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    dataiter = iter(train_loader)
    x, y = dataiter.next()
    # print(x, x.shape)
    # print(y, y.shape)

    best_accuracy = -10.0
    best_model_id = -1
    df = pd.DataFrame([[None, None]], columns=['model id', 'accuracy'])
    for model_id, Net in enumerate(models):
        net = Net.to(device)
        criterion = nn.MSELoss()
        optimizer = optim.SGD(params=net.parameters(), lr=LEARNING_RATE, momentum=0.5)

        if to_train:
            path = train(net, train_loader, criterion, optimizer, device, model_id)
        if to_test:
            accuracy = test(net, test_loader, device, path, model_id, dis_threshold)
            record = [[model_id, accuracy]]
            df_record = pd.DataFrame(record, columns=['model id', 'accuracy'])
            df = df.append(df_record)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_id = model_id

            df.to_csv('record.csv', index=False)
            print("best accuracy: %.3f%%" % (100 * best_accuracy))
            print("best model id: %d" % best_model_id)
        if test_trained_models:
            root_path = './model_data/'
            file_name = str(model_id) + '.pth'
            path = root_path + file_name
            accuracy = test(net, test_loader, device, path, model_id, TEST_THRESHOLD)
            

if __name__ == "__main__":
    main(to_train=False, to_test=False, test_trained_models=True)