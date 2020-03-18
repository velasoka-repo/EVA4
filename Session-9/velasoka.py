import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import MultiStepLR
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


def cifar10_dataset(transform=transforms, train=False):
    """
    :param train:
    :param transform:
    :return: CIFAR10  datasets
    """
    return datasets.CIFAR10(root="./data", train=train, transform=transforms, download=True)


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

    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)


def model_summary(model, input_size, device):
    """
    shows `Module` summary. It shows parameter for all Convolution fn like Conv2d, MaxPooling, ReLU etc
    :param model:
    :param input_size:
    :param device:
    """
    model = model.to(device)
    summary(model=model, input_size=input_size)


def sgd_optimizer(model, lr=0.001, momentum=0.9):
    """
    :param model:
    :param lr: default lr=0.001
    :param momentum: default momentum=0.9
    :return: `StochasticGradientDescent` optimizer
    """
    return optim.SGD(model.parameters(), lr=lr, momentum=momentum)


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
            f"Training Batch={batch_id}, loss={loss.item():.5f}, Correct Prediction={correct}/{total}, Train Accuracy={100 * correct / total:.5f}")


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
        train_cnn(model, data_loader[0], loss_fn, optimizer, device)
        test_cnn(model, data_loader[1], device)

        if scheduler:
            scheduler.step()


def drive_path(folder, filename):
    """
    :param folder:
    :param filename:
    :return: Google Drive file path
    """
    return f"/content/gdrive/My Drive/Colab Notebooks/EVA4/{folder}/{filename}"


def save_model(model, path):
    """
    save the model state
    :param model:
    :param path:
    :return:
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


def multisteplr_schedular(optimizer, milestone=[10, 20], gamma=0.1):
    """
    :param optimizer:
    :param milestone: default value is [10, 20]
    :param gamma: default value is 0.1
    :return: MultiStepLR
    """
    return MultiStepLR(optimizer=optimizer, milestones=milestone, gamma=gamma)


def steplr_schedular(optimizer, step_size=10, gamma=0.1):
    """
    :param optimizer:
    :param step_size: default value is 10
    :param gamma: default value is 0.1
    :return:
    """
    return StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)


def show_image(np_arr):
    """
    :param np_arr: numpy array (height, width, channels)
    """
    np_arr = np_arr.astype(np.uint8)
    plt.imshow(np_arr)
    plt.show()


def show_gradcam_image(img, label, gradcam_img, predicted_label):
    """
    :param img: numpy array
    :param label: cifar10 class label index
    :param gradcam_img: numpy array
    :param predicted_label: cifar10 class label index
    """
    fig, (axis1, axis2) = plt.subplots(1, 2)
    axis1.imshow(img.astype(np.uint8))
    axis1.set_title(f"Actual Label: `{label}`")
    axis2.imshow(gradcam_img.astype(np.uint8))
    axis2.set_title(f"Predicted Label: `{predicted_label}`")
    plt.show()


def show_gradcam_visualize(cifar10_labels, data_list):
    """
    :param cifar10_labels: cifar10 class labels
    :param data_list:
    """
    for (img, label, gradcam_img, pred_label) in data_list:
        r_label = cifar10_labels[label]
        p_label = cifar10_labels[pred_label]
        show_gradcam_image(img, r_label, gradcam_img, p_label)


def denormalize_tensor(t_arr):
    """
    :param t_arr: normalized(0.5) tensor value
    :return normalized numpy array
    """
    tmp = t_arr.squeeze()
    tmp = (tmp / 2 + 0.5) * 255  # de-normalize tensor
    tmp = tmp.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)  # convert float to int
    return tmp
