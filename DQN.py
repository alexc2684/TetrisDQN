import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=2, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=2, stride=1)
        self.fc1 = nn.Linear(1536, 256)
        self.fc2 = nn.Linear(256, 7)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        return self.fc2(x)
