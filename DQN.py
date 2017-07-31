import pytorch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.hidden = nn.Linear(201, 100)
        self.fc = nn.Linear(100, 7)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        return self.fc(x)
