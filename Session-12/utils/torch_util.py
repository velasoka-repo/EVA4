import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import torch
import numpy as np
from torchsummary import summary


def optimizer():
    return optim


def scheduler():
    return lr_scheduler


def loss_fns():
    return nn


def model_summary(model, input_size):
    """
    :param model:
    :param input_size:
    :param device:
    """
    summary(model=model, input_size=input_size)


def device():
    """
    :return: available device (CUDA | CPU)
    """
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(device=device_name)


def drive_path(filename, folder=None):
    """
    :param folder:
    :param filename:
    :return: Google Drive file path
    """
    if folder is None:
        return filename

    return f"/content/gdrive/My Drive/Colab Notebooks/EVA4/{folder}/{filename}"


def save_model(model, path):
    """
    save the model state
    :param model:
    :param path:
    """
    torch.save(model.state_dict(), path)


def load_model(path, map_location=None):
    """
    load the trained model
    :param map_location:
    :param path:
    :return: model stat_dict
    """
    if map_location:
        return torch.load(path, map_location)

    return torch.load(path)


def normalized_mean_std(mean, std, range=255):
    """
    :param range: default 255
    :param mean: 0-255 range
    :param std: 0-255 range
    :return: (mean, std) of 0-1 range
    """
    m = np.array(mean) / range
    s = np.array(std) / range
    return m, s
