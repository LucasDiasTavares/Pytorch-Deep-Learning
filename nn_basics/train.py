import warnings
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from model import Model
from data import load_data
from evaluate import Evaluate


warnings.simplefilter(action='ignore', category=FutureWarning)  # ignore future warning

random_seed = 32
torch.manual_seed(random_seed)
model = Model()

path = "nn_data/"
file_name = "iris.csv"
df_without_variety_train, df_without_variety_test, df_variety_train, df_variety_test = load_data(path + file_name)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

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

# Save Model
print("Saving Model ...")
model_name = 'iris_flower_model.pt'
torch.save(model.state_dict(), path + model_name)

evaluator = Evaluate()
print("Saving evaluation_loss.log")
evaluator.evaluate_loss(model, criterion, df_without_variety_test, df_variety_test)

print("Saving evaluation_predictions.log")
evaluator.evaluate_predictions(model, df_without_variety_test, df_variety_test)

# Create Graph
plt.plot(range(epochs), losses)
plt.ylabel("loss/error")
plt.xlabel("Epoch")
plt.title('Graph')
plt.show()
