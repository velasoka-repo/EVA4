import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import torch
import numpy as np
from torch_lr_finder import LRFinder


def optimizer():
    return optim


def scheduler():
    return lr_scheduler


def loss():
    return nn


def device():
    """
    :return: available device (CUDA | CPU)
    """
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(device=device_name)


def lr_finder(model, optimizer, criterion, device):
    return LRFinder(model=model, optimizer=optimizer, criterion=criterion, device=device)


def get_triangler_lr(iteration, step_size, min_lr, max_lr):
    """Given the inputs, calculates the lr that should be applicable for this iteration"""
    cycle = np.floor(1 + iteration / (2 * step_size))
    x = np.abs(iteration / step_size - 2 * cycle + 1)
    lr = min_lr + (max_lr - min_lr) * np.maximum(0, (1 - x))
    return lr


class OneCyclePolicy:
    def __init__(self, optimizer, step_size, min_lr, max_lr):
        self.optimizer = optimizer
        self.step_size = step_size
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_tend = []
        self.lr_tend.append(self.get_lr())
        self.iteration_trend = []
        self.iteration_trend.append(0)
        self.iteration = 0
        self.default_lr = self.get_lr()
        self.max_reached = False

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def step(self):
        self.iteration += 1
        for param_group in self.optimizer.param_groups:
            tmp = param_group['lr']
            if tmp:
                if tmp >= self.max_lr:
                    self.max_reached = True

                if self.max_reached:
                    param_group['lr'] = self.default_lr
                else:
                    self.iteration_trend.append(self.iteration)
                    new_lr = get_triangler_lr(self.iteration, self.step_size, self.min_lr, self.max_lr)
                    param_group['lr'] = new_lr
                    self.lr_tend.append(new_lr)
                    print(f"LR value updated from {tmp} to {new_lr}")

    def trends(self):
        return self.lr_tend, self.iteration_trend


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
