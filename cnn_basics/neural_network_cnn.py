import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader  # https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
from torchvision import datasets, transforms
from utils import create_cnn_data_folder

cnn_data = create_cnn_data_folder()

transform = transforms.ToTensor()

# Download MINIST https://www.tensorflow.org/datasets/catalog/mnist?hl=pt-br
train_data = datasets.MNIST(root=cnn_data, train=True, download=True, transform=transform)
test_data = datasets.MNIST(root=cnn_data, train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10, shuffle=False)


# Model
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(5*5*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, data):
        data = F.relu(self.conv1(data))
        data = F.max_pool2d(data, 2, 2)
        data = F.relu(self.conv2(data))
        data = F.max_pool2d(data, 2, 2)

        data = data.view(-1, 5*5*16)

        data = F.relu(self.fc1(data))
        data = F.relu(self.fc2(data))
        data = self.fc3(data)

        return F.log_softmax(data, dim=1)


# Instance
torch.manual_seed(41)
model = Model()

# Loss
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



















