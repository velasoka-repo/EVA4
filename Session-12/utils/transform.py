import torchvision.transforms as tf


def torch_transform():
    return tf


def compose_transforms(*args):
    """
    :param args: list of transforms to be applied on Image
    :return: Compose([args])
    """
    return tf.Compose(list(args))
