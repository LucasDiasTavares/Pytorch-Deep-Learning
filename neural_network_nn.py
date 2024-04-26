import pdb
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt  # https://matplotlib.org
import pandas as pd
from sklearn.model_selection import train_test_split

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


# Manual seed
random_seed = 41
torch.manual_seed(random_seed)
model = Model()


path = "nn_data/iris.csv"
df = pd.read_csv(path)

df['variety'] = df['variety'].replace('Setosa', 0.0)
df['variety'] = df['variety'].replace('Versicolor', 1.0)
df['variety'] = df['variety'].replace('Virginica', 2.0)

# Remove variety, because it's the results itself
df_without_variety = df.drop('variety', axis=1)
df_variety = df['variety']

# Convert to arrays
df_without_variety = df_without_variety.values
df_variety = df_variety.values

# Train Test Split - sklearn
df_without_variety_train, df_without_variety_test, df_variety_train, df_variety_test = train_test_split(
    df_without_variety, df_variety, test_size=0.2, random_state=random_seed)

# Convert arrays to tensors
# X
df_without_variety_train = torch.FloatTensor(df_without_variety_train)
df_without_variety_test = torch.FloatTensor(df_without_variety_test)

# Y
df_variety_train = torch.LongTensor(df_variety_train)
df_variety_test = torch.LongTensor(df_variety_test)

# Model measure errors
# Predictions
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train Model
epochs = 100
losses = []

for epo in range(epochs):
    # Get prediction
    variety_prediction = model.forward(df_without_variety_train)

    # Measure loss
    loss = criterion(variety_prediction, df_variety_train)

    # Check losses manually
    losses.append(loss.detach().numpy())

    if epo % 10 == 0:
        print(f"Epoch: {epo} and loss: {loss}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.plot(range(epochs), losses)
plt.ylabel("loss/error")
plt.xlabel("Epoch")
plt.title('Graph')
plt.show()
