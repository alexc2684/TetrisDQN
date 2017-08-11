import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=2, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(2736, 2000)
        self.fc2 = nn.Linear(2000, 1000)
        self.fc3 = nn.Linear(1000, 200)
        self.fc4 = nn.Linear(200, 7)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.fc1(x.view(x.size(0), -1))
        x = self.fc2(x)
        x = self.fc3(x)
        return self.fc4(x)
