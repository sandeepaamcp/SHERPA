from src.dataset.datasetStrategy import poison_strategy_with_non_iid_split
from util import constants as const

import torch

from util.constants import DATASET_ROOT

torch.cuda.current_device()
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10, GTSRB
from torchvision.datasets import MNIST
from torchvision.datasets import FashionMNIST
from torch.utils.data import Subset
from src.dataset.customDatasets.NSL_KDD import generate_nsl_kdd_dataset
from src.dataset.customDatasets.NIDD_5G import generate_5g_nidd_dataset
from src.dataset.customDatasets.CelebA import generate_celeba_dataset


def get_classes():
    # CLASSES_CIFAR10 = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
    # CLASSES_MNIST = ('1','2','3','4','5','6','7','8','9','0')
    CLASSES_MNIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    CLASSES_FMNIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    CLASSES_CIFAR10 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    CLASSES_NSL_KDD_ALL = [0, 1, 2, 3, 4]
    CLASSES_NSL_KDD = [0, 1]  # ONLY SELECT DOS ATTACKS AND NORMAL TRAFFIC
    CLASSES_5G_NIDD = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    CLASSES_5G_NIDD_TEST = [0, 1]
    CLASSES_CELEBA = [0, 1]
    CLASSES_GTSRB = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,
                     33,34,35,36,37,38,39,40,41,42]


    # print(const.SELECTED_DATASET)
    if const.SELECTED_DATASET == 'MNIST':
        return CLASSES_MNIST
    elif const.SELECTED_DATASET == 'FMNIST':
        return CLASSES_FMNIST
    elif const.SELECTED_DATASET == 'CIFAR-10':
        return CLASSES_CIFAR10
    elif const.SELECTED_DATASET == 'NSL-KDD':
        return CLASSES_NSL_KDD
    elif const.SELECTED_DATASET == 'NSL-KDD-ALL':
        return CLASSES_NSL_KDD_ALL
    elif const.SELECTED_DATASET == '5G-NIDD':
        return CLASSES_5G_NIDD_TEST
    elif const.SELECTED_DATASET == 'CELEBA':
        return CLASSES_CELEBA
    elif const.SELECTED_DATASET == 'GTSRB':
        return CLASSES_GTSRB
    else:
        raise ValueError("Invalid dataset name")


def get_dataset(len_train=10000, len_test=1000, all_data=False):
    if const.SELECTED_DATASET == 'MNIST':
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
        )
        trainset_all = MNIST(DATASET_ROOT + "/dataset", train=True, download=True, transform=transform)
        testset_all = MNIST(DATASET_ROOT + "/dataset", train=False, download=True, transform=transform)
    elif const.SELECTED_DATASET == 'FMNIST':
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
        )
        trainset_all = FashionMNIST(DATASET_ROOT + "/dataset", train=True, download=True, transform=transform)
        testset_all = FashionMNIST(DATASET_ROOT + "/dataset", train=False, download=True, transform=transform)
    elif const.SELECTED_DATASET == 'CIFAR-10':
        transform_train = transforms.Compose(
            [transforms.Resize((32, 32)),  # resises the image so it can be perfect for our model.
             transforms.RandomHorizontalFlip(),  # FLips the image w.r.t horizontal axis
             transforms.RandomRotation(10),  # Rotates the image to a specified angel
             transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
             # Performs actions like zooms, change shear angles.
             transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Set the color params
             transforms.ToTensor(),  # comvert the image to tensor so that it can work with torch
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize all the images
             ])
        transform_cifar_test = transforms.Compose([transforms.Resize((32, 32)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])

        trainset_all = CIFAR10(DATASET_ROOT + "/dataset", train=True, download=True, transform=transform_train)
        testset_all = CIFAR10(DATASET_ROOT + "/dataset", train=False, download=True, transform=transform_cifar_test)
    elif const.SELECTED_DATASET == 'NSL-KDD':
        trainset_all, testset_all = generate_nsl_kdd_dataset(is_only_dos=True)
    elif const.SELECTED_DATASET == 'NSL-KDD-ALL':
        trainset_all, testset_all = generate_nsl_kdd_dataset(is_only_dos=False)
    elif const.SELECTED_DATASET == '5G-NIDD':
        trainset_all, testset_all = generate_5g_nidd_dataset()
    elif const.SELECTED_DATASET == 'CELEBA':
        trainset_all, testset_all = generate_celeba_dataset()
    elif const.SELECTED_DATASET == 'GTSRB':
        transform = transforms.Compose([transforms.Resize([32, 32]), transforms.ToTensor()])
        trainset_all = GTSRB(DATASET_ROOT + "/dataset", split='train', download=True, transform=transform)
        testset_all = GTSRB(DATASET_ROOT + "/dataset", split='test', download=True, transform=transform)
    else:
        raise ValueError("Invalid dataset name")

    subset_train_indices = range(len_train)
    trainset = Subset(trainset_all, subset_train_indices)

    subset_test_indices = range(len_test)
    testset = Subset(testset_all, subset_test_indices)
    if all_data:
        return trainset_all, testset_all
    else:
        return trainset, testset


def get_testloader(len_test, batch_size, shuffle=True):

    subset_test_indices = range(len_test)

    if const.SELECTED_DATASET == 'MNIST':
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
        )
        testset_all = MNIST(DATASET_ROOT + "/dataset", train=False, download=False, transform=transform)
    elif const.SELECTED_DATASET == 'FMNIST':
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
        )
        testset_all = FashionMNIST(DATASET_ROOT + "/dataset", train=False, download=False, transform=transform)

    elif const.SELECTED_DATASET == 'CIFAR-10':
        transform_cifar_test = transforms.Compose([transforms.Resize((32, 32)),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                   ])
        testset_all = CIFAR10(DATASET_ROOT + "/dataset", train=False, download=False, transform=transform_cifar_test)

    elif const.SELECTED_DATASET == 'NSL-KDD':
        _, testset_all = generate_nsl_kdd_dataset(is_only_dos=True)

    elif const.SELECTED_DATASET == 'NSL-KDD-ALL':
        _, testset_all = generate_nsl_kdd_dataset(is_only_dos=False)

    elif const.SELECTED_DATASET == '5G-NIDD':
        _, testset_all = generate_5g_nidd_dataset()

    elif const.SELECTED_DATASET == 'CELEBA':
        _, testset_all = generate_celeba_dataset(only_test=True)

    elif const.SELECTED_DATASET == 'GTSRB':
        transform = transforms.Compose([transforms.Resize([32, 32]), transforms.ToTensor()])
        testset_all = GTSRB(DATASET_ROOT + "/dataset", split='test', download=False, transform=transform)

    else:
        raise ValueError("Invalid dataset name")

    testset = Subset(testset_all, subset_test_indices)  # iid testset
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=shuffle)
    return testloader


# iid load dataset function
def iid_split(len_train, len_test):
    # Download and transform CIFAR-10 (train and test)
    # transform = transforms.Compose(
    #     [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    # )

    # Split training set into 10 partitions to simulate the individual dataset
    trainset, testset = get_dataset(len_train, len_test)
    partition_size = len(trainset) // const.NUM_CLIENTS
    lengths = [partition_size] * const.NUM_CLIENTS
    datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))

    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for ds in datasets:
        len_val = len(ds) // 10  # 10 % validation set
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
        trainloaders.append(DataLoader(ds_train, batch_size=const.BATCH_SIZE, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=const.BATCH_SIZE))
    testloader = DataLoader(testset, batch_size=const.BATCH_SIZE)
    return trainloaders, valloaders, testloader


def non_iid_split(len_train, len_test, split_strategy, label_list, random_ratio, **kwargs):
    trainset, testset = get_dataset(all_data=True)
    partition_size = len_train // const.NUM_CLIENTS
    lengths = [partition_size] * const.NUM_CLIENTS

    datasets = split_strategy(trainset, lengths, const.NUM_CLIENTS, label_list, random_ratio, **kwargs)

    # Split each partition into train/val and create DataLoader
    train_loaders = []
    val_loaders = []
    for ds in datasets:
        len_val = len(ds) // 10  # 10 % validation set
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
        train_loaders.append(DataLoader(ds_train, batch_size=const.BATCH_SIZE, shuffle=True))
        val_loaders.append(DataLoader(ds_val, batch_size=const.BATCH_SIZE))
    test_loader = DataLoader(testset, batch_size=const.BATCH_SIZE)
    return train_loaders, val_loaders, test_loader


def non_iid_train_val_separate_split(len_train, len_test, split_strategy, label_list, random_ratio, kwargs_train,
                                     kwargs_val):
    trainset, testset = get_dataset(all_data=True)  # we get all dataset since split strategy may require fixed amount
    # of data for each class, which is not known in advance
    partition_size = len_train // const.NUM_CLIENTS
    lengths = [partition_size] * const.NUM_CLIENTS

    print('generating train set')
    datasets_train = split_strategy(trainset, lengths, const.NUM_CLIENTS, label_list, random_ratio, **kwargs_train)

    partition_size_val = len_test // const.NUM_CLIENTS
    lengths_val = [partition_size_val] * const.NUM_CLIENTS
    print('generating validation set')
    datasets_val = split_strategy(testset, lengths_val, const.NUM_CLIENTS, label_list, random_ratio, **kwargs_val)

    # Split each partition into train/val and create DataLoader
    train_loaders = []
    val_loaders = []
    for i in range(const.NUM_CLIENTS):
        train_loaders.append(DataLoader(datasets_train[i], batch_size=const.BATCH_SIZE, shuffle=True))
        val_loaders.append(DataLoader(datasets_val[i], batch_size=const.BATCH_SIZE))
    test_loader = DataLoader(testset, batch_size=const.BATCH_SIZE)
    return train_loaders, val_loaders, test_loader
