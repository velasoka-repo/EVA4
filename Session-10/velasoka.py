import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

SEED = 6


def available_device():
    """
    :return: either `cuda` or `cpu` device based on availability
    """
    name = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(name)


def to_normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    """
    normalize `Tensor` with mean & std value

    (i.e) normalize = (each pixel - mean) / std
    :param mean:
    :param std:
    :return: `transforms.Normalize(mean, std)`
    """
    return transforms.Normalize(mean=mean, std=std)


def to_tensor():
    """
    converts `numpy` | `PILImage` to `Tensor`
    :return: `transforms.ToTensor()`
    """
    return transforms.ToTensor()


def to_pil():
    """
    converts `Tensor` to `PILImage`
    :return: `transforms.ToPILImage()`
    """
    return transforms.ToPILImage()


def compose_transform(*args):
    """
    compose multiple transforms

    (i.e) transforms.ToTensor(), transforms.Normalize(mean=0.5, std=0.5)
    :param args: args...
    :return: `transforms.Compose([...])`
    """
    return transforms.Compose(list(args))


def cifar10_dataset(transform, train=False):
    """
    :param train:
    :param transform:
    :return: CIFAR10  datasets
    """
    return datasets.CIFAR10(root="./data", train=train, transform=transform, download=True)


def cifar10_dataloader(dataset, batch_size=4, shuffle=False):
    """
    :param shuffle:
    :param train_dataset:
    :param batch_size: default batch_size=4
    :return: CIFAR10 data_loader
    """
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=6)


def model_summary(model, input_size, device):
    """
    shows `Module` summary. It shows parameter for all Convolution fn like Conv2d, MaxPooling, ReLU etc
    :param model:
    :param input_size:
    :param device:
    """
    model = model.to(device)
    summary(model=model, input_size=input_size)


def sgd_optimizer(model, lr=0.001, momentum=0.9, weight_decay=0, nesterov=False):
    """
    :param weight_decay: default 0
    :param nesterov: default False
    :param model:
    :param lr: default lr=0.001
    :param momentum: default momentum=0.9
    :return: `StochasticGradientDescent` optimizer
    """
    return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)


def ce_loss():
    """
    :return: CrossEntropyLoss
    """
    return nn.CrossEntropyLoss()


def train_cnn(model, data_loader, loss_fn, optimizer, device):
    """
    model training code with `model.train()` enabled
    :param model:
    :param data_loader:
    :param loss_fn:
    :param optimizer:
    :param device:
    """
    model.train()
    correct = 0
    total = 0
    pbar = tqdm(iterable=data_loader, total=len(data_loader), position=0)
    for batch_id, (data, label) in enumerate(pbar):
        data, label = data.to(device), label.to(device)

        optimizer.zero_grad()  # clear previously calculated gradient

        output = model(data)

        loss = loss_fn(output, label)

        # back propagation & update weight
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)  # choose max element index
        correct += pred.eq(label.view_as(pred)).sum().item()
        total += len(data)

        pbar.set_description(
            f"Training Batch={batch_id}, loss={loss.item():.5f}, Correct Prediction={correct}/{total}, Train Accuracy={100. * correct / total:.5f}")

    return float(f"{100. * correct / total:.5f}")


def test_cnn(model, data_loader, device):
    """
    model test code with `model.eval()` enabled
    :param model:
    :param data_loader:
    :param device:
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(iterable=data_loader, total=len(data_loader), position=0)
        for batch_id, (data, label) in enumerate(pbar):
            data, label = data.to(device), label.to(device)

            output = model(data)

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()
            total += len(data)
            pbar.set_description(
                f"Test Batch={batch_id}, Correct Validation={correct}/{total}, Test Accuracy={100 * correct / total:.5f}")

    return float(f"{100. * correct / total:.5f}")


def start_cnn(model, optimizer, loss_fn, data_loader, device, epochs=10, scheduler=None):
    """
    start CNN with `train` & `test`
    :param model:
    :param optimizer:
    :param loss_fn:
    :param data_loader: tuple of (train, test) data_loader
    :param device:
    :param epochs:
    :param scheduler:
    """
    model = model.to(device)
    for epoch in range(1, epochs + 1):
        print(f"\nEPOCH: {epoch}")
        train_accuracy = train_cnn(model, data_loader[0], loss_fn, optimizer, device)
        test_accuracy = test_cnn(model, data_loader[1], device)

        if float(train_accuracy) == 100:
            print(f"Train Accuracy Reached 100%. Terminating epoch at {epoch}")
            break

        if scheduler:
            scheduler.step(test_accuracy)


def drive_path(filename, folder=None):
    """
    :param folder:
    :param filename:
    :return: Google Drive file path
    """
    if folder is None:
        return filename

    return f"/content/gdrive/My Drive/Colab Notebooks/EVA4/{folder}/{filename}"


def plot_accuracy_or_loss(points, label=None, xlabel=None, ylabel=None):
    """
    :param points: test, train accuracy | loss
    :param label:
    """
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.plot(points)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if label:
        axis.set_title(label)
    plt.show()


def save_model(model, path):
    """
    save the model state
    :param model:
    :param path:
    """
    torch.save(model.state_dict(), path)


def save_last_run(run_id, path):
    """
    :param run_id: int to be stored
    :param path:
    """
    torch.save(torch.tensor([run_id]), path)


def get_last_run(path="last_run.pt"):
    """
    :param path: default last_run.pt
    :return: last run id
    """
    return load_model(path).item()


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


def multisteplr_scheduler(optimizer, milestone=[10, 20], gamma=0.1):
    """
    :param optimizer:
    :param milestone: default value is [10, 20]
    :param gamma: default value is 0.1
    :return: MultiStepLR
    """
    return MultiStepLR(optimizer=optimizer, milestones=milestone, gamma=gamma)


def steplr_scheduler(optimizer, step_size=10, gamma=0.1):
    """
    :param optimizer:
    :param step_size: default value is 10
    :param gamma: default value is 0.1
    :return:
    """
    return StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)


def reduce_lr_scheduler(optimizer, mode='min', factor=0.1, patience=0, verbose=True, min_lr=0.000001):
    """
    :param min_lr: default 0.000001 (don't go lr value below this range)
    :param optimizer:
    :param mode: default 'min'
    :param factor: default 0.1
    :param patience: default 0 (once loss is greater than previous, decrease the lr value)
    :param verbose: default True
    :return: ReduceLROnPlateau
    """
    return ReduceLROnPlateau(optimizer=optimizer, mode=mode, factor=factor, patience=patience, verbose=verbose,
                             min_lr=min_lr)


def show_image(np_arr):
    """
    :param np_arr: numpy array (height, width, channels)
    """
    np_arr = np_arr.astype(np.uint8)
    plt.imshow(np_arr)
    plt.show()


def pil_to_numpy(pil):
    """
    :param pil: PILImage
    :return: numpy array
    """
    return np.asarray(pil)


def random_images(dataset, count=25):
    """
    :param dataset: train | test dataset without any transformation
    :param count: default is 25
    :return: dict of (images, labels) =  images & labels length is equal to count
    """
    import random
    random.seed(SEED)
    img_list = []
    label_list = []
    data = {}
    for i in range(count):
        index = random.randrange(0, len(dataset))
        pil_img, label = dataset[index]
        img = pil_to_numpy(pil_img)
        img_list.append(img)
        label_list.append(label_list)
    return img_list, label_list


def mean_and_std(dataset):
    """
    :param dataset: train | test dataset without any transformation
    :return: mean, std
    """
    mean = np.mean(dataset.data, axis=(0, 1, 2))
    std = np.std(dataset.data, axis=(0, 1, 2))
    return mean, std


def show_2image(img1, img2, img1_label=None, img2_label=None):
    """
    :param img1: numpy array
    :param img1_label: cifar10 class label index
    :param img2: numpy array
    :param img2_label: cifar10 class label index
    """
    fig, (axis1, axis2) = plt.subplots(1, 2)
    axis1.imshow(img1.astype(np.uint8))
    if img1_label:
        axis1.set_title(img1_label)
    axis2.imshow(img2.astype(np.uint8))
    if img2_label:
        axis2.set_title(img2_label)
    plt.show()


def show_images(data_list, rows, columns):
    """
    :param rows:
    :param columns:
    :param data_list: must be this format [(text, img, label), ......]
    """
    if len(data_list) < (rows * columns):
        print(f"Invalid len of data_list {len(data_list)} is must be >= with rows*columns {rows * columns}")
        return

    fig, axis = plt.subplots(nrows=rows, ncols=columns, constrained_layout=True, figsize=(columns * 2.5, rows * 2.5))
    idx = 0
    for row_axis in axis:
        for column_axis in row_axis:
            (img, label) = data_list[idx]
            column_axis.imshow(img.astype(np.uint8))
            if label:
                column_axis.set_title(label)
            column_axis.axis("off")
            idx += 1
    plt.show()


def denormalize_tensor(t_arr, mean=0.5, std=0.5, max_pixel=255):
    """
    :param max_pixel: default RGB max value is 255
    :param std: default 0.5
    :param mean: default 0.5
    :param t_arr: normalized(0.5) tensor value
    :return normalized numpy array
    """
    tmp = t_arr.squeeze()
    tmp = (tmp * mean + std) * max_pixel  # de-normalize tensor
    tmp = tmp.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)  # convert float to int
    return tmp
