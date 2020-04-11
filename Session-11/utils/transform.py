import torchvision.transforms as tf
import albumentations as al


def torch_transform():
    return tf


def album_transform():
    return al


def compose_transforms(*args):
    """
    :param args: list of transforms to be applied on Image
    :return: Compose([args])
    """
    return tf.Compose(list(args))
