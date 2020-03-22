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


def tensor():
    """
    :return: albumentations Tensor
    """
    return album_torch.ToTensor()


def normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    """
    :param mean:
    :param std:
    :return: albumentations Normalize
    """
    return album.Normalize(mean=mean, std=std)


def flip(p=0.5):
    """
    :param p: probability of applying the transform.
    :return: albumentations Flip
    """
    return album.Flip(p=p)


def hue_satur_value(hue=172, satur=20, val_shift=27, p=0.5):
    """
    :param hue:
    :param satur:
    :param val_shift:
    :param p: probability of applying the transform.
    :return: albumentations HueSaturationValue
    """
    return album.HueSaturationValue(hue_shift_limit=hue, sat_shift_limit=satur, val_shift_limit=val_shift, p=p)


def compression(quality_lower=4, quality_upper=100, p=0.5):
    """
    :param quality_lower:
    :param quality_upper:
    :param p: probability of applying the transform.
    :return: albumentations JpegCompression
    """
    return album.JpegCompression(quality_lower=quality_lower, quality_upper=quality_upper, p=p)


def blur(blur_limit=5, p=0.5):
    """
    :param blur_limit:
    :param p: probability of applying the transform.
    :return: albumentations Blur
    """
    return album.Blur(blur_limit=blur_limit, p=p)


def gray():
    """
    :return: albumentations ToGray()
    """
    return album.ToGray()


def channel_shuffle(p=0.5):
    """
    :param p: probability of applying the transform.
    :return: albumentations ChannelShuffle
    """
    return album.ChannelShuffle(p=p)


def gaussnoise(p=0.5):
    """
    :param p: probability of applying the transform.
    :return: albumentations GaussNoise
    """
    return album.GaussNoise(p=p)


def hflip(p=0.5):
    """
    :param p: probability of applying the transform.
    :return: albumentations HorizontalFlip
    """
    return album.HorizontalFlip(p=p)


def rotate(limit=286, p=0.5):
    """
    :param limit: range from which a random angle is picked
    :param p: probability of applying the transform.
    :return: albumentations Rotate
    """
    return album.Rotate(limit=limit, p=p)


def rgb_shift(r_shift=105, g_shift=45, b_shift=40, p=0.5):
    """
    :param b_shift: range for changing values for the blur channel.
    :param g_shift: range for changing values for the green channel.
    :param r_shift: range for changing values for the red channel.
    :param p: probability of applying the transform.
    :return: albumentations RGBShift
    """
    return album.RGBShift(r_shift_limit=r_shift, g_shift_limit=g_shift, b_shift_limit=b_shift, p=p)


def randomgamma(gamma=(80, 120), p=0.5):
    """
    :param gamma: default (80, 120)
    :param p: probability of applying the transform.
    :return: albumentations RandomGamma
    """
    return album.RandomGamma(gamma_limit=gamma, p=p)


def vflip(p=0.5):
    """
    :param p: probability of applying the transform.
    :return: albumentations VerticalFlip
    """
    return album.VerticalFlip(p=p)


def cutout(num_holes=1, max_h_size=10, max_w_size=10, fill_value=0, p=0.5):
    """
    :return: albumentations Cutout
    """
    return album.Cutout(num_holes=num_holes, max_h_size=max_h_size, max_w_size=max_w_size, fill_value=fill_value, p=p)


def random_rotate90(p=0.5):
    """
    :param p: probability of applying the transform.
    :return: albumentations RandomRotate90
    """
    return album.RandomRotate90(p=p)


def random_brightness(brightness=0.2, contrast=0.2, p=0.5):
    """
    :param contrast: default 0.2
    :param brightness: default 0.2
    :param p: probability of applying the transform.
    :return: albumentations RandomBrightnessContrast
    """
    return album.RandomBrightnessContrast(brightness_limit=brightness, contrast_limit=contrast, p=p)


def compose(*args):
    """
    :param args: arguments of transformation
    :return: albumentations.Compose(...
    """
    return album.Compose(transforms=list(args))


class Dataset(Dataset):
    def __init__(self, data, labels, transform=None):
        """
        :param data:
        :param labels:
        :param transform:
        """
        self.data = data
        self.labels = labels
        self.transform = transform

        if self.transform:
            tmp = []
            for image in data:
                # pre-process the transformation to calculate mean and std
                augmented = self.transform(image=image)  # it has only ['image']
                image = augmented['image']
                tmp += [image]
            self.data = torch.stack(tmp)  # convert list to tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        image = self.data[item]
        label = self.labels[item]
        return image, label


def transform_dataset(data, labels, *args):
    """
    :param labels:
    :param data:
    :param *arg: provide required transformations
    :return: Albumentations Dataset
    """
    if len(args):
        transform = compose(*args)
    else:
        transform = compose(normalize(), tensor())

    return Dataset(data=data, labels=labels, transform=transform)


def augment_image(transform, img):
    """
    :param transform: transformation to be applied on the image
    :param img: image to apply arg transform
    :return: augmented image
    """
    return transform(image=img)['image']


def data_transform_list(mean):
    """
    :param mean: mean value of the dataset
    :return:
    """
    return [
        ("Cutout", cutout(fill_value=mean, p=1)),
        ("HorizontalFlip", hflip(p=1)),
        ("VerticalFlip", vflip(p=1)),
        ("Brightness", random_brightness(p=1, brightness=0.3, contrast=0.2)),
        ("Gamma", randomgamma(p=1, gamma=(10, 100))),
        ("RGBShift", rgb_shift(p=1)),
        ("Rotate", rotate(p=1)),
        ("RandomRotate", random_rotate90(p=0.6)),
        ("Flip", flip(p=0.8)),
        ("Gray", gray()),  # not applied
        ("ChannelShuff", channel_shuffle(p=1)),  # calling method has channels_shuffle=[0, 1, 2]
        ("GaussNoise", gaussnoise(p=1)),
        ("HueSatValue", hue_satur_value(p=1)),
        ("JPEGComp", compression(p=1)),
        ("Blur", blur(p=1))]


# TODO
def mean_std(*args):
    """
    :param args: transformations to calculate mean & std
    :return: mean and std of R, G, B channels
    """
    dataset = transform_dataset(*args)
    mean_r = torch.mean(dataset.data[:, 0:1, :, :]).item()
    mean_g = torch.mean(dataset.data[:, 1:2, :, :]).item()
    mean_b = torch.mean(dataset.data[:, 2:3, :, :]).item()
    std_r = torch.std(dataset.data[:, 0:1, :, :]).item()
    std_g = torch.std(dataset.data[:, 1:2, :, :]).item()
    std_b = torch.std(dataset.data[:, 2:3, :, :]).item()
    return (mean_r, mean_g, mean_b), (std_r, std_g, std_b)
