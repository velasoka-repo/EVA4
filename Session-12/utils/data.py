import time

import torch
import torchvision.transforms as tf
import torchvision.datasets as ds
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# ../../tiny-imagenet-200/train/n03992509/images/n03992509_0.JPEG

"""
DataLoader Args:
             batch_size: int = ...,
             shuffle: bool = ...,
             sampler: Optional[Sampler[int]] = ...,
             num_workers: int = ...,
             collate_fn: (List[T]) -> Any = ...,
             pin_memory: bool = ...,
             drop_last: bool = ...,
             timeout: float = ...,
             worker_init_fn: (int) -> None = ...
"""


class LoadImage:
    def __init__(self, root_path, size=500):
        """
        :param root_path: `root` path from where to load TinyImageNet dataset
        """
        file = f"{root_path}/wnids.txt"
        dir_names = []
        dir_labels = {}
        with open(file, mode="r") as file_reader:
            for dir_name in file_reader.readlines():
                dir_names.append(dir_name.replace("\n", ""))

        # file = f"{data_path}/words.txt"
        # with open(file, mode="r") as file_reader:
        #     for line in file_reader.readlines():
        #         dir_name, label = line.split(sep="\t", maxsplit=2)
        #         dir_labels[dir_name] = label.replace("\n", "")

        train_data, train_labels = [], []
        test_data, test_labels = [], []
        all_data = []
        N = 350
        start_time = time.time()
        for image_label, dir_name in enumerate(dir_names):
            for index in range(size):
                path = f"{root_path}/train/{dir_name}/images/{dir_name}_{index}.JPEG"
                pil = Image.open(path)
                pil = pil if pil.mode == "RGB" else pil.convert(mode="RGB")
                img_data = np.asarray(pil)
                if index < N:
                    train_data.append(img_data)
                    train_labels.append(image_label)
                else:
                    test_data.append(img_data)
                    test_labels.append(image_label)
                all_data.append(img_data)
        end_time = time.time()
        print(f"Total time taken to load image: {(end_time - start_time):.2f} seconds")

        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.mean = np.mean(all_data, axis=(0, 1, 2))
        self.std = np.std(all_data, axis=(0, 1, 2))
        self.data_path = root_path

    def __str__(self):
        return f"Train dataset: {len(self.train_data)}\nTest dataset: {len(self.test_data)}\nMean: {self.mean}\nStd: {self.std}\nData path: {self.data_path}"


class ImageDataset(Dataset):
    def __init__(self, dataset, labels, transforms=None):
        """
        :param dataset: train | test numpy array dataset
        :param labels: labels associated with dataset
        :param transforms: default None
        """
        self.dataset = dataset
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        label = self.labels[item]
        if self.transforms:
            data = self.transforms(data)
        return data, label


class TinyImageNetDataLoader:
    def __init__(self, dataset, batch_size=100, shuffle=False, **kwargs):
        """
        :param dataset: train | test dataset
        :param batch_size: default 100
        :param shuffle: default False
        :param kwargs: {'pin_memory':true, 'num_workers':4, ......}
        """
        self.dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, **kwargs);

    def loader(self):
        return self.dataloader


if __name__ == '__main__':
    # TinyImageNetDataLoader(data_path="../../tiny-imagenet-200")
    data = LoadImage(root_path="../../tiny-imagenet-200")
    print(data)
    mean = np.array(data.mean) / 255
    std = np.array(data.std) / 255
    print(mean, std)
    c = tf.Compose([tf.ToTensor()])
    d = data.train_data[0]
    print(type(d))
    d = c(d)
    print(torch.max(d), torch.min(d))
