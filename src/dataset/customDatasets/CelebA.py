import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as tfms
from torch.utils.data import Subset

from util.constants import DATASET_ROOT


class CustomCelebA(datasets.CelebA):
    def __init__(self, *args, target_attr_name=None, transform=None,
                 target_transform=None, **kwargs):
        super(CustomCelebA, self).__init__(*args, transform=transform,
                                           target_transform=target_transform, **kwargs)
        self.target_attr_name = target_attr_name

    def __getitem__(self, index):
        img, target = super(CustomCelebA, self).__getitem__(index)

        # Filter the attributes to include only the desired one(s)
        # if self.target_attr_names is not None:
        #     target_indices = [self.attr_names.index(attr) for attr in self.target_attr_names]
        #     target = target[target_indices]

        if self.target_attr_name is not None:
            target = target[self.attr_names.index(self.target_attr_name)]

        return img, target



class CustomCelebAMulti(datasets.CelebA):
    def __init__(self, *args, target_attr_names=None, transform=None, target_transform=None, **kwargs):
        super(CustomCelebAMulti, self).__init__(*args, transform=transform, target_transform=target_transform, **kwargs)
        self.target_attr_names = target_attr_names

    def __getitem__(self, index):
        img, target = super(CustomCelebAMulti, self).__getitem__(index)

        # Filter the attributes to include only the desired ones
        if self.target_attr_names is not None:
            target_indices = [self.attr_names.index(attr) for attr in self.target_attr_names]
            target = target[target_indices]
            # target = target_one_hot.index(1)

        return img, target

def generate_celeba_dataset(train_examples=10000, test_examples=1000, only_test=False):
    target_attribute = "Male"
    # target_attributes = ['Black_Hair', 'Blond_Hair','Brown_Hair', 'Gray_Hair', 'Straight_Hair', 'Wavy_Hair']
    # target_attributes = ['Black_Hair', 'Blond_Hair','Brown_Hair']

    image_size = 32
    imagenet_mean = [0.485, 0.456, 0.406]  # mean of the ImageNet dataset for normalizing
    imagenet_std = [0.229, 0.224, 0.225]  # std of the ImageNet dataset for normalizing

    transforms = tfms.Compose(
        [
            tfms.Resize((image_size, image_size)),
            tfms.ToTensor(),
            tfms.Normalize(imagenet_mean, imagenet_std),
        ]
    )
    test_dataset = CustomCelebA(
        DATASET_ROOT + '/dataset',
        split="test",
        target_attr_name=target_attribute,
        transform=transforms,
        download=False,  # Set to True if dataset needs to be downloaded
        # limit = test_examples  # Set the limit here
    )
    if only_test:
        return None, test_dataset

    train_dataset = CustomCelebA(
        DATASET_ROOT+'/dataset',
        split="train",
        target_attr_name=target_attribute,
        transform=transforms,
        download=False,  # Set to True if dataset needs to be downloaded
        # limit = train_examples  # Set the limit here
    )

    return Subset(train_dataset, np.arange(1, train_examples)), Subset(test_dataset, np.arange(1, test_examples))
