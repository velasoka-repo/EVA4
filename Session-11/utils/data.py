import torchvision.datasets as ds
from torch.utils.data import DataLoader


class CIFAR10:
    def __init__(self, data_path="../data", train_transform=None, test_transform=None):
        """
        :param data_path: CIFAR10 image downloading path (default ../data)
        :param train_transform: transform to be applied for train dataset
        :param test_transform: transform to be applied for test dataset
        """
        self.train_data = ds.CIFAR10(root=data_path, train=True, download=True, transform=train_transform)
        self.test_data = ds.CIFAR10(root=data_path, train=False, download=True, transform=test_transform)

    def dataloader(self, batch_size=100, **kwargs):
        """
        :param batch_size: default 100
        :param kwargs: num_workers, pin_memory etc
        :return: tuple of (train & test dataloader)
        """
        train_dataloader = self.train_dataloader(batch_size=batch_size, **kwargs)
        test_dataloader = self.test_dataloader(batch_size=batch_size, **kwargs)
        return train_dataloader, test_dataloader

    def train_dataloader(self, batch_size=100, **kwargs):
        """
        :param batch_size: default 100
        :param kwargs: num_workers, pin_memory etc
        :return:
        """
        return DataLoader(dataset=self.train_data, batch_size=batch_size, shuffle=True, **kwargs)

    def test_dataloader(self, batch_size=100, **kwargs):
        """
        :param batch_size: default 100
        :param kwargs: num_workers, pin_memory etc
        :return:
        """
        return DataLoader(dataset=self.test_data, batch_size=batch_size, shuffle=False, **kwargs)


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
if __name__ == '__main__':
    pass
