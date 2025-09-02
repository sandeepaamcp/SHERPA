import random

from matplotlib import pyplot as plt
from torch.utils.data import Subset, DataLoader
from torchvision.datasets import MNIST, CIFAR10

from src.dataset.datasetHandler import get_classes, non_iid_train_val_separate_split
from src.dataset.datasetStrategy import poison_strategy_with_non_iid_split
import torchvision.transforms as transforms
from src.dataset.datasetHandler import generate_nsl_kdd_dataset
from src.dataset.datasetHandler import generate_5g_nidd_dataset

from util.constants import SELECTED_DATASET, DATASET_ROOT

# train_loaders = None
# val_loaders = None
# test_loaders = None


class ClientConfig:
    train_loaders = None
    val_loaders = None
    test_loaders = None

    def __init__(self, cid):
        self.cid = cid
        self.train_loaders = None
        self.val_loaders = None
        self.test_loaders = None

    def set_trainloader(self, trainloader):
        self.train_loaders = trainloader

    def set_val_loader(self, valloader):
        self.val_loaders = valloader

    def get_trainloader(self):
        if self.train_loaders is None:
            raise ValueError("train loaders cannot be None")
        return self.train_loaders

    def get_valloader(self):
        if self.val_loaders is None:
            raise ValueError("val loaders cannot be None")
        return self.val_loaders


def generate_data_loaders(kwargs_train, kwargs_val,
                          split_mechanism=non_iid_train_val_separate_split,
                          strategy=poison_strategy_with_non_iid_split,
                          len_train_data=10000, len_test_data=1000,
                          random_ratio=1, is_visualize=False,
                          visualize_idx=0):
    label_list = get_classes()
    print(len_train_data)
    # kwargs_train = {'poison_type':'random_poison','poison_ratio':1,'target_clients':[1,2,3,4,5]}
    # kwargs_train = {'poison_type': 'target_poison', 'poison_ratio': 1, 'target_label': 4, 'target_clients': [1, 2, 3]}
    # kwargs_val = {'poison_type': 'random_poison', 'poison_ratio': 0, 'target_clients': []}
    # kwargs = {'poison_type':'random_poison','poison_ratio':0,'target_clients':[]} kwargs = {
    # 'poison_type':'random_poison','poison_ratio':0.5,'target_clients':[1, 2, 3, 4, 5, 6, 7, 8]} trainloaders,
    # valloaders, testloader = non_iid_split(10000, 1000, poison_strategy_with_non_iid_split, label_list, 0.1,
    # **kwargs) # non iid data 90%. However looks like the clustering method fails for targeted poison trainloaders,
    # valloaders, testloader = non_iid_split(10000, 1000, poison_strategy_with_non_iid_split, label_list, 1, **kwargs)

    # trainloaders, valloaders, testloader = non_iid_split(10000, 1000, poison_strategy_with_non_iid_split,
    # label_list, 1, **kwargs)

    trainloader, valloader, testloader = split_mechanism(len_train_data, len_test_data,
                                                         strategy,
                                                         label_list,
                                                         random_ratio,
                                                         kwargs_train,
                                                         kwargs_val)

    if is_visualize:
        subset_dataset = trainloader[visualize_idx].dataset
        visualize_data_distribution(subset_dataset)

    return trainloader, valloader, testloader


# def get_train_val_test_loaders():
#     if train_loaders is None or val_loaders is None or test_loaders is None:
#         raise ValueError("data loaders cannot be None")
#
#     return train_loaders, val_loaders, test_loaders


def visualize_data_distribution(subset_dataset):
    # Define the number of random labels to visualize
    num_samples = 5
    # Generate random indices for the samples
    random_indices = random.sample(range(len(subset_dataset)), num_samples)
    # Plot the random labels
    fig, axes = plt.subplots(1, num_samples, figsize=(12, 3))

    for i, idx in enumerate(random_indices):
        image, label = subset_dataset[idx]
        axes[i].imshow(image[0], cmap='gray')
        axes[i].set_title(f"Label: {label}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

