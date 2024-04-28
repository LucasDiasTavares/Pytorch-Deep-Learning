import torch
from model import Model
from utils import convert_number_to_class


def predict(model, data):
    with torch.no_grad():
        prediction = model(data)
        return prediction


if __name__ == "__main__":
    model = Model()
    model.load_state_dict(torch.load("nn_data/iris_flower_model.pt"))
    new_data = torch.tensor([6.2, 3.4, 5.4, 2.3])  # Virginica
    #           0         1         2
    # tensor([-8.5095,  2.0114,  6.6947])
    # code thinks its number 2 and it's correct!
    prediction = predict(model, new_data)
    flower_name = convert_number_to_class(prediction.argmax().item())
    print(f"Prediction: {prediction}\nPredicted Class: {prediction.argmax().item()}\nFlower Name: {flower_name}")
