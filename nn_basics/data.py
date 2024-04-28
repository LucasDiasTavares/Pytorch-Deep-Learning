import pandas as pd
from sklearn.model_selection import train_test_split
import torch


def load_data(file_path, random_seed=32, test_size=0.2):
    df = pd.read_csv(file_path)

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
        df_without_variety, df_variety, test_size=test_size, random_state=random_seed)

    # Convert arrays to tensors
    df_without_variety_train = torch.FloatTensor(df_without_variety_train)
    df_without_variety_test = torch.FloatTensor(df_without_variety_test)
    df_variety_train = torch.LongTensor(df_variety_train)
    df_variety_test = torch.LongTensor(df_variety_test)

    return df_without_variety_train, df_without_variety_test, df_variety_train, df_variety_test
