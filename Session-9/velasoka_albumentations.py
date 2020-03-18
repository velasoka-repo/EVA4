import numpy as np
import torch
import albumentations as album
import albumentations.pytorch as album_torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset
import pickle

DATA_PATH = "./data/cifar-10-batches-py"


def unbind_data(path):
    with open(path, 'rb') as fo:  # rb == read binary
        dict = pickle.load(fo, encoding='bytes')
    return dict


def read_metadata(path):
    m_path = f"{path}/batches.meta"
    return read_data(m_path)  # dict_keys([b'num_cases_per_batch', b'label_names', b'num_vis'])


def cifar10_class_names():
    """
    :return: tuple of CIFAR10 class names
    """
    dic = read_metadata(DATA_PATH)
    return tuple([str(name, 'utf-8') for name in dic[b'label_names']])


def read_data(path):
    """
    :param path:
    :return: dictionary of data with labels
    """
    return unbind_data(path)


def reformat_data(dic):
    """
    reshape the flatten array to [3x32x32] and apply transpose in data
    :param dic:
    :return: restructed data, labels
    """
    # dic has [b'batch_label', b'labels', b'data', b'filenames']
    data, labels = dic[b'data'], dic[b'labels']

    # data[i] has 3072 value in flatten array (3x32x32)
    data = data.reshape(len(data), 3, 32, 32)

    # TODO: do I need 3x32x32 to be changed into 32x32x3 ?
    # yes, otherwise its failing with ValueError: operands could not be broadcast together with shapes (3,32,32) (3,) (3,32,32)

    data = data.transpose(0, 2, 3, 1)  # transposing [1000, 32, 32, 3] to  [1000, 32, 32, 3]

    return data, labels


def read_cifar10_train_data(path=DATA_PATH, index=5):
    """
    :param path:
    :param index:
    :return: train_data
    """
    train_data = None
    train_labels = []
    for i in range(1, index + 1):  # default 5 train dataset available in CIFAR10
        t_path = f"{path}/data_batch_{i}"
        dic = read_data(t_path)

        # dic has [b'batch_label', b'labels', b'data', b'filenames']
        data, labels = reformat_data(dic)

        train_data = data if i == 1 else np.vstack((train_data, data))  # vstack adds new data to existing array
        train_labels += labels

    return train_data, np.array(train_labels)


def read_cifar10_test_data(path=DATA_PATH):
    """
    :param path:
    :return: test_data
    """
    t_path = f"{path}/test_batch"
    dic = read_data(t_path)

    # dic has [b'batch_label', b'labels', b'data', b'filenames']
    data, labels = reformat_data(dic)

    return data, np.array(labels)


def al_tensor():
    """
    :return: albumentations Tensor
    """
    return album_torch.ToTensor()


def al_normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    """
    :param mean:
    :param std:
    :return: albumentations Normalize
    """
    return album.Normalize(mean=mean, std=std)


def al_hflip(p=0.5):
    """
    :param p: probability of applying the transform.
    :return: albumentations HorizontalFlip
    """
    return album.HorizontalFlip(p=0.5)


def al_rotate(limit=286, p=0.5):
    """
    :param limit: range from which a random angle is picked
    :param p: probability of applying the transform.
    :return: albumentations Rotate
    """
    return album.Rotate(limit=286, p=0.5)


def al_rgb_shift(r_shift=105, g_shift=45, b_shift=40, p=0.5):
    """
    :param b_shift: range for changing values for the blur channel.
    :param g_shift: range for changing values for the green channel.
    :param r_shift: range for changing values for the red channel.
    :param p: probability of applying the transform.
    :return: albumentations RGBShift
    """
    return album.RGBShift(r_shift_limit=105, g_shift_limit=45, b_shift_limit=40, p=0.5)


def al_randomgamma(p=0.5):
    """
    :param p: probability of applying the transform.
    :return: albumentations RandomGamma
    """
    return album.RandomGamma(p=0.5)


def al_vflip(p=0.5):
    """
    :param p: probability of applying the transform.
    :return: albumentations VerticalFlip
    """
    return album.VerticalFlip(p=0.5)


def al_random_rotate90(p=0.5):
    """
    :param p: probability of applying the transform.
    :return: albumentations RandomRotate90
    """
    return album.RandomRotate90(p=0.5)


def al_random_brightness( p=0.5):
    """
    :param p: probability of applying the transform.
    :return: albumentations RandomBrightnessContrast
    """
    return album.RandomBrightnessContrast(p=0.5)


def al_compose(*args):
    """
    :param args: arguments of transformation
    :return: albumentations.Compose(...
    """
    return album.Compose(transforms=list(args))


class AlbumentationDataset(Dataset):
    def __init__(self, data, labels, transforms=None):
        """
        :param data:
        :param labels:
        :param transforms:
        """
        self.data = data
        self.labels = labels
        self.transforms = transforms

        tmp = []
        for image in data:
            # pre-process the transformation to calculate mean and std
            if self.transforms:
                augmented = self.transforms(image=image)  # it has only ['image']
                image = augmented['image']
            tmp += [image]
        self.data = torch.stack(tmp)  # convert list to tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        image = self.data[item]
        label = self.labels[item]
        return image, label


def albumentation_dataset(data, labels, *args):
    """
    :param labels:
    :param data:
    :param *arg: provide required transformations
    :return: Albumentations Dataset
    """
    if len(args):
        transforms = al_compose(*args)
    else:
        transforms = al_compose(al_hflip(), al_tensor())

    return AlbumentationDataset(data=data, labels=labels, transforms=transforms)


def mean_std(*args):
    """
    :param args: transformations to calculate mean & std
    :return: mean and std of R, G, B channels
    """
    dataset = albumentation_dataset(*args)
    mean_r = torch.mean(dataset.data[:, 0:1, :, :]).item()
    mean_g = torch.mean(dataset.data[:, 1:2, :, :]).item()
    mean_b = torch.mean(dataset.data[:, 2:3, :, :]).item()
    std_r = torch.std(dataset.data[:, 0:1, :, :]).item()
    std_g = torch.std(dataset.data[:, 1:2, :, :]).item()
    std_b = torch.std(dataset.data[:, 2:3, :, :]).item()
    return (mean_r, mean_g, mean_b), (std_r, std_g, std_b)
