from torch_lr_finder import LRFinder


def finder(model, optimizer, loss_fn, device):
    """
    :param model:
    :param optimizer: SGD with momentum
    :param loss_fn: any of the loss function NLL, CrossEntroyLoss, L1, L2 etc
    :param device: cuda | cpu
    :return:
    """
    return LRFinder(model=model, optimizer=optimizer, criterion=loss_fn, device=device)


def train_range_test(lr_finder, train_loader, end_lr=100, num_itr=100, step_mode="exp"):
    """
    :param lr_finder: LRFinder from torch_lr_finder
    :param train_loader: train dataloader
    :param end_lr: default 100
    :param num_itr: default 100
    :param step_mode: exponential
    """
    lr_finder.range_test(train_loader=train_loader, end_lr=end_lr, num_iter=num_itr, step_mode=step_mode)


def test_range_test(lr_finder, train_loader, test_loader, end_lr=100, num_itr=100, step_mode="exp"):
    """
       :param test_loader: test dataloader
       :param lr_finder: LRFinder from torch_lr_finder
       :param train_loader: train dataloader
       :param end_lr: default 100
       :param num_itr: default 100
       :param step_mode: exponential
       """
    lr_finder.range_test(train_loader=train_loader, val_loader=test_loader, end_lr=end_lr, num_iter=num_itr,
                         step_mode=step_mode)


def plot_loss(lr_finder):
    """
    :param lr_finder: LRFinder from torch_lr_finder
    """
    lr_finder.plot()


def reset(lr_finder):
    """
    :param lr_finder: LRFinder from torch_lr_finder
    """
    lr_finder.reset()


def history(lr_finder):
    """
    :param lr_finder: LRFinder from torch_lr_finder
    :return: history of LRFinder
    """
    return lr_finder.history
