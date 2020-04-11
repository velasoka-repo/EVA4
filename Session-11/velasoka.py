import utils.data as data
import utils.augment as aug
import utils.transform as tf
import model.nn as nn
import utils.visualize as view
import utils.torch_util as util
import utils.runner as network


def cifar10_dataloader(train_transform, test_transform, batch_size=512):
    kwargs = {"num_workers": 4, "pin_memory": True}

    # CIFAR10 dataset
    cifar10 = data.CIFAR10(data_path="./data", train_transform=train_transform, test_transform=test_transform)
    train_dataloader = cifar10.train_dataloader(batch_size=batch_size, **kwargs)
    test_dataloader = cifar10.test_dataloader(batch_size=100, **kwargs)

    return train_dataloader, test_dataloader


def cifar10(batch_size=512):
    torch_tf = tf.torch_transform()

    # mean for CIFAR10 dataset R=125.30691918, G=122.95038918, B=113.86545097
    # (125.30691918, 122.95038918, 113.86545097)

    # Augmentation
    crop = torch_tf.RandomCrop(size=32, padding=4, padding_mode="edge")
    fliplr = aug.Fliplr()
    cutout = aug.Cutout(max_hw=(8, 8), fill_value=(125.30691918, 122.95038918, 113.86545097))

    # Transform
    to_tensor = torch_tf.ToTensor()
    normalize = torch_tf.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    train_transform = tf.compose_transforms(to_tensor, normalize)
    test_transform = tf.compose_transforms(to_tensor, normalize)

    original_data = cifar10_dataloader(train_transform=train_transform, test_transform=test_transform,
                                       batch_size=batch_size)

    train_transform = tf.compose_transforms(crop, fliplr, cutout, to_tensor, normalize)
    augmented_data = cifar10_dataloader(train_transform=train_transform, test_transform=test_transform,
                                        batch_size=batch_size)

    return original_data[0], augmented_data[0], augmented_data[1]
