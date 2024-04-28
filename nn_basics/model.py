import warnings
import torch.nn as nn
import torch.nn.functional as F


warnings.simplefilter(action='ignore', category=FutureWarning)  # ignore future warning


class Model(nn.Module):
    # Input 4 features of the flower
    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
        super().__init__()  # instantiate
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    # Moving Forward
    def forward(self, data):
        # h1
        data = F.relu(self.fc1(data))
        # h1 to h2
        data = F.relu((self.fc2(data)))
        # output
        data = self.out(data)  # F.softmax(self.out(data), dim=1)  # Apply softmax activation
        return data
