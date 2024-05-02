import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader  # https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from decouple import config
from sklearn.metrics import confusion_matrix
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
        self.fc1 = nn.Linear(5 * 5 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, data):
        data = F.relu(self.conv1(data))
        data = F.max_pool2d(data, 2, 2)
        data = F.relu(self.conv2(data))
        data = F.max_pool2d(data, 2, 2)

        data = data.view(-1, 5 * 5 * 16)

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

# Train and Test
import time

# Training `1.3544928352038066 minutes
start = time.time()

epochs = 5
train_losses = []
test_losses = []
train_correct = []
test_correct = []

for epo in range(epochs):
    train_crt = 0
    test_crt = 0

    for idx, (x_train, y_train) in enumerate(train_loader):
        idx += 1
        y_pred = model(x_train)  # predict from training 2D
        loss = criterion(y_pred, y_train)  # compare predictions

        predicted = torch.max(y_pred.data, 1)[1]
        batch_correct = (predicted == y_train).sum()
        train_crt += batch_correct

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 600 == 0:
            print(f"Epoch: {epo} - Batch: {idx} - Loss: {loss.item()}")

    train_losses.append(loss)
    train_correct.append(train_crt)

    with torch.no_grad():
        for b, (x_test, y_test) in enumerate(test_loader):
            y_val = model(x_test)
            predicted = torch.max(y_val.data, 1)[1]
            test_crt += (predicted == y_test).sum()

    loss = criterion(y_val, y_test)
    test_losses.append(loss)
    test_correct.append(test_crt)

current = time.time()
total = current - start
print(f"Training `{total / 60} minutes")
# Train and Test

# Graph Loss
train_losses = [tl.item() for tl in train_losses]
plt.plot(train_losses, label="Training")
plt.plot(test_losses, label="Validation")
plt.title("Loss")
plt.legend()
plt.show()

# Graph Accuracy
plt.plot([t / 600 for t in train_correct], label="Training")
plt.plot([t / 100 for t in test_correct], label="Validation")
plt.title("Accuracy")
plt.legend()
plt.show()

test_load_everything = DataLoader(test_data, batch_size=10000, shuffle=False)

with torch.no_grad():
    correct = 0
    for X_test, y_test in test_load_everything:
        y_val = model(X_test)
        predicted = torch.max(y_val, 1)[1]
        correct += (predicted == y_test).sum()
    # 98.68%
    print(f"Percent of correct: {correct.item() / len(test_data) * 100}%")

# Grab an image
# test_data[1978]

# Grab data inside image
# test_data[1978][0]

# Reshape it
# test_data[1978][0].reshape(28, 28)

# Show the image number 4
plt.imshow(test_data[1978][0].reshape(28, 28))
plt.show()

# Pass the image thru our model
model.eval()
with torch.no_grad():
    new_prediction = model(test_data[1978][0].view(1, 1, 28, 28))

# tensor(4)
print(new_prediction.argmax())

