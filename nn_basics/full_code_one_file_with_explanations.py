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
random_seed = 32
torch.manual_seed(random_seed)
model = Model()

path = "nn_data/"
file_name = "iris.csv"
df = pd.read_csv(path+file_name)

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

# Testing and Evaluating

# Evaluate Model
with torch.no_grad():
    y_eval = model.forward(df_without_variety_test)
    loss = criterion(y_eval, df_variety_test)
    print(loss)

# Check if predictions is correct
correct = 0
with torch.no_grad():
    for idx, data in enumerate(df_without_variety_test):
        y_eval = model.forward(data)
        # Show result from my code thinks it is correct
        # 1 - tensor([-5.8771,  4.4629,  6.5155]) - 2 - 2
        # Inside tensor probability
        # -5.8771 means Setosa (number 0)
        # 4.4629 means Versicolor (number 1)
        # 6.5155 means Virginica (number 2)
        # Highest number is what flower its choose
        # First 2 means the number of flower
        # Second 2 correct number of flower
        print(f"{idx + 1} - {str(y_eval)} - {df_variety_test[idx]} - {y_eval.argmax().item()}")

        if y_eval.argmax().item() == df_variety_test[idx]:
            correct += 1

    print(f"Corrects: {correct}")

# Add New Data
# 6.2,3.4,5.4,2.3,"Virginica"
# Setosa == 0
# Versicolor == 1
# Virginica == 2

new_data = torch.tensor([6.2, 3.4, 5.4, 2.3])
with torch.no_grad():
    #           0         1         2
    # tensor([-8.5095,  2.0114,  6.6947])
    # code thinks its number 2 and it's correct!
    print(f"New Data: {model(new_data)}")


# Save Model
model_name = 'iris_flower_model.pt'
torch.save(model.state_dict(), path+model_name)
