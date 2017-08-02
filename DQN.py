import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.hidden1 = nn.Linear(213, 2000)
        self.hidden2 = nn.Linear(2000, 1000)
        self.hidden3 = nn.Linear(1000, 500)
        self.fc = nn.Linear(500, 7)


    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        return self.fc(x)
