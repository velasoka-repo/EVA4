import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm

SEED = 6

def get_available_device():
    """
    :return: either `cuda` or `cpu` device based on availability
    """
    name = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(name)


def to_normalize(mean, std):
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


def get_cifar10_dataset(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    """
    :param mean:
    :param std:
    :return: CIFAR10 (train, test) datasets, the dataset is applied with transforms like `ToTensor` & `Normalize`
    """
    compose_trans = compose_transform(to_tensor(), to_normalize(mean, std))
    train_dataset = datasets.CIFAR10(root="./data", train=True, transform=compose_trans, download=True)
    test_dataset = datasets.CIFAR10(root="./data", train=False, transform=compose_trans, download=True)
    return train_dataset, test_dataset


def get_cifar10_data_loader(dataset, batch_size=4):
    """
    :param dataset: tuple of (train, test) dataset
    :param batch_size:
    :return: CIFAR10 (train, test) data_loader
    """
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        
    train_data_loader = DataLoader(dataset=dataset[0], batch_size=batch_size, shuffle=True, num_workers=2)
    test_data_loader = DataLoader(dataset=dataset[1], batch_size=batch_size, shuffle=False, num_workers=2)
    return train_data_loader, test_data_loader


def model_summary(model, input_size, device):
    """
    shows `Module` summary. It shows parameter for all Convolution fn like Conv2d, MaxPooling, ReLU etc
    :param model:
    :param input_size:
    :param device:
    """
    model = model.to(device)
    summary(model=model, input_size=input_size)


def get_sgd_optimizer(model, lr=0.001, momentum=0.9):
    """
    :param model:
    :param lr:
    :param momentum:
    :return: `StochasticGradientDescent` optimizer with default `lr=0.001, momentum=0.9`
    """
    return optim.SGD(model.parameters(), lr=lr, momentum=momentum)


def get_ce_loss():
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

        pbar.set_description(f"Training Batch={batch_id}, loss={loss.item():.5f}, Correct Prediction={correct}/{total}, Train Accuracy={100 * correct / total:.5f}")


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
            pbar.set_description(f"Test Batch={batch_id}, Correct Validation={correct}/{total}, Test Accuracy={100 * correct / total:.5f}")


def save_model(model, path):
    """
    save the model state
    :param model:
    :param path:
    :return:
    """
    torch.save(model.state_dict(), path)


def start_cnn(model, optimizer, loss_fn, data_loader, device, epochs=10):
    """
    start CNN with `train` & `test`
    :param model:
    :param optimizer:
    :param loss_fn:
    :param data_loader: tuple of (train, test) data_loader
    :param device:
    :param epochs:
    """
    model = model.to(device)
    for epoch in range(1, epochs):
        print(f"\nEPOCH: {epoch}")
        train_cnn(model, data_loader[0], loss_fn, optimizer, device)
        test_cnn(model, data_loader[1], device)


def get_drive_path(folder, filename):
    """
    :param folder:
    :param filename:
    :return: Google Drive file path
    """
    return f"/content/gdrive/My Drive/Colab Notebooks/EVA4/{folder}/{filename}"
