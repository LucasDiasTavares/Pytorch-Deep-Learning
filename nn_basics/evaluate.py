import torch
from utils import create_output_file


class Evaluate:
    @staticmethod
    def evaluate_loss(model, criterion, df_without_variety_test, df_variety_test):
        create_output_file('evaluation_loss.log')
        with torch.no_grad(), open('nn_data/evaluation_loss.log', 'w') as f:
            predictions = model.forward(df_without_variety_test)
            loss = criterion(predictions, df_variety_test)
            f.write(f"Test Loss: {loss.item()}")

    @staticmethod
    def evaluate_predictions(model, df_without_variety_test, df_variety_test):
        correct = 0
        create_output_file('evaluation_predictions.log')
        with torch.no_grad(), open('nn_data/evaluation_predictions.log', 'w') as f:
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
                output_str = f"{idx + 1} - {str(y_eval)} - {df_variety_test[idx]} - {y_eval.argmax().item()}\n"
                f.write(output_str)

                if y_eval.argmax().item() == df_variety_test[idx]:
                    correct += 1

            f.write(f"Corrects: {correct}\n")
