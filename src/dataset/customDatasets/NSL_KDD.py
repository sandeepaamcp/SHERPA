import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.dataset.customDatasets.CustomDataset import CustomDataset
from util.constants import DATASET_ROOT


# Helper function to generate NSL-KDD dataset. The normalized and categorized sample of training data csv is provided
# separately and can be found in link bit.ly/46jexfU. Path to dataset should be adjusted

# Create a custom dataset class to wrap your data
def generate_nsl_kdd_dataset(train_examples=100000, test_examples=10000, is_only_dos=True):
    file_path_full_training_set = (DATASET_ROOT + '/dataset/NSL-KDD'
                                   '/normalized_and_categorized_training_data.csv')

    columns = (
    ['duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
     'num_compromised', 'root_shell',
     'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
     'is_host_login', 'is_guest_login', 'count',
     'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
     'srv_diff_host_rate', 'dst_host_count',
     'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
     'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
     'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'level', 'protocol_type_cat',
     'service_cat', 'flag_cat', 'attack_map'])

    df = pd.read_csv(file_path_full_training_set, header=None, names=columns)
    if is_only_dos:
        df2 = df.loc[df.iloc[:, -1] < 2]
    else:
        df2 = df.copy()
    # randomize dataset
    df2 = df2.sample(frac=1, random_state=42).reset_index(drop=True)

    y = df2.iloc[:, -1:].values.flatten()
    vals, count = np.unique(y, return_counts=True)
    print(vals)
    num_classes = len(vals)
    df2.drop(df.columns[len(df.columns) - 1], axis=1, inplace=True)
    x = df2.values
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
