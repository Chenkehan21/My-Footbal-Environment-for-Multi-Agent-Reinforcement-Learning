import torch
import torch.nn as nn
import numpy as np

class DQN(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(DQN, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_features=input_shape, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=output_shape)
        )

    def forward(self, x):
        # x = x.double()
        # x = x.view(x.size[0], -1)

        return self.fc(x)