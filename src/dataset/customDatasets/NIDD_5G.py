import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from src.dataset.customDatasets.CustomDataset import CustomDataset
from util.constants import DATASET_ROOT


def generate_5g_nidd_dataset(train_examples=10000, test_examples=1000):
    file_path_full_training_set = (DATASET_ROOT + '/dataset/5G-NIDD'
                                   '/Combined_processed.csv')
    df = pd.read_csv(file_path_full_training_set)

    x = df.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df2 = pd.DataFrame(x_scaled)
    df2[df2.columns[-1]] = df[df.columns[-1]]
    df2.columns = df.columns

    # randomize dataset
    df3 = df2.copy().loc[df.iloc[:, -1] < 2] # get only the highest 2 available classes
    df3 = df3.sample(frac=1, random_state=42).reset_index(drop=True)

    y = df3.iloc[:, -1:].values.flatten()
    vals, count = np.unique(y, return_counts=True)
    print(vals)
    num_classes = len(vals)
    df3.drop(df.columns[len(df.columns) - 1], axis=1, inplace=True)
    x = df3.values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
    x_train = x_train[:train_examples]
    y_train = y_train[:train_examples]
    x_test = x_test[:test_examples]
    y_test = y_test[:test_examples]
    print('Train samples:', len(x_train))
    print('Test samples:', len(x_test))

    # Create instances of the custom dataset for train and test sets
    train_dataset = CustomDataset(x_train, y_train)
    test_dataset = CustomDataset(x_test, y_test)

    return train_dataset, test_dataset
