import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.hidden1 = nn.Linear(213, 213)
        self.hidden2 = nn.Linear(213, 150)
        self.fc = nn.Linear(150, 6)


    def forward(self, x):
        x = F.tanh(self.hidden1(x))
        x = F.tanh(self.hidden2(x))
        return self.fc(x)
