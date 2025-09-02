import torch

torch.cuda.current_device()

import random

from torch.utils.data import Subset


def random_swap(dataset_samp, poison_ratio):
    dataset_size = len(dataset_samp)
    thresh = int(dataset_size * poison_ratio)
    s1 = random.sample(range(0, int(dataset_size / 2)), int(thresh / 2))
    s2 = random.sample(range(int(dataset_size / 2), dataset_size), int(thresh / 2))

    list_ds = []
    # for i in range(len(dataset_samp)):
    #     list_ds.append([dataset_samp[i][0], dataset_samp[i][-1]])
    for i in range(len(dataset_samp)):
        if type(dataset_samp[i][-1]) == torch.Tensor:
            list_ds.append([dataset_samp[i][0].clone().detach(), dataset_samp[i][-1].clone().detach()])
        else:
            list_ds.append([dataset_samp[i][0].clone().detach(), dataset_samp[i][-1]])

    for i in range(len(s1)):
        i1 = s1[i]
        i2 = s2[i]
        y1 = list_ds[i1][-1]
        y2 = list_ds[i2][-1]
        list_ds[i1][-1] = y2
        list_ds[i2][-1] = y1
    subset_new = Subset(list_ds, [i for i in range(len(list_ds))])
    return subset_new


def random_poison(dataset_samp, poison_ratio, label_list):
    dataset_size = len(dataset_samp)
    thresh = int(dataset_size * poison_ratio)
    s1 = random.sample(range(0, int(dataset_size)), int(thresh))

    list_ds = []
    # for i in range(len(dataset_samp)):
    #     list_ds.append([dataset_samp[i][0], dataset_samp[i][-1]])
    for i in range(len(dataset_samp)):
        if type(dataset_samp[i][-1]) == torch.Tensor:
            list_ds.append([dataset_samp[i][0].clone().detach(), dataset_samp[i][-1].clone().detach()])
        else:
            list_ds.append([dataset_samp[i][0].clone().detach(), dataset_samp[i][-1]])

    for i in range(len(s1)):
        i1 = s1[i]
        y1 = list_ds[i1][-1]
        y2 = label_list[random.choice(range(len(label_list)))]
        list_ds[i1][-1] = y2
        # list_ds[i2][-1]= y1
    subset_new = Subset(list_ds, [i for i in range(len(list_ds))])
    return subset_new


def target_poison(dataset_samp, poison_ratio, target_label):
    dataset_size = len(dataset_samp)
    thresh = int(dataset_size * poison_ratio)
    s1 = random.sample(range(0, int(dataset_size)), int(thresh))

    list_ds = []
    for i in range(len(dataset_samp)):
        #   list_ds.append([dataset_samp[i][0],dataset_samp[i][-1]])
        if type(dataset_samp[i][-1]) == torch.Tensor:
            list_ds.append([dataset_samp[i][0].clone().detach(), dataset_samp[i][-1].clone().detach()])
        else:
            list_ds.append([dataset_samp[i][0].clone().detach(), dataset_samp[i][-1]])

    for i in range(len(s1)):
        i1 = s1[i]
        list_ds[i1][-1] = target_label

    subset_new = Subset(list_ds, [i for i in range(len(list_ds))])
    return subset_new

def target_poison_new(dataset_samp, poison_ratio, target_label):
    dataset_size = len(dataset_samp)
    thresh = int(dataset_size * poison_ratio)
    s1 = random.sample(range(0, int(dataset_size)), int(thresh))

    list_ds = []
    for i in range(len(dataset_samp)):
        #   list_ds.append([dataset_samp[i][0],dataset_samp[i][-1]])
        if type(dataset_samp[i][-1]) == torch.Tensor:
            list_ds.append([dataset_samp[i][0].clone().detach(), dataset_samp[i][-1].clone().detach()])
        else:
            list_ds.append([dataset_samp[i][0].clone().detach(), dataset_samp[i][-1]])

    for i in range(len(s1)):
        i1 = s1[i]
        list_ds[i1][-1] = target_label

    subset_new = Subset(list_ds, [i for i in range(len(list_ds))])
    return subset_new