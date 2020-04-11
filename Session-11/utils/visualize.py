import matplotlib.pyplot as plt


def show_image(img):
    """
    :param img: PIL.Image | numpy array
    """
    plt.imshow(img)
    plt.show()


def show_graph(x_value, y_value, x_label, y_label, title):
    fig, ax = plt.subplots()
    ax.plot(x_value, y_value)
    ax.set(xlabel=f'{x_label}', ylabel=f'{y_label}',
           title=f'{title}')
    ax.grid()
    plt.show()


import numpy as np


def get_triangler_lr(iteration, step_size, min_lr, max_lr):
    cycle = np.floor(1 + iteration / (2 * step_size))
    x = np.abs(iteration / step_size - 2 * cycle + 1)
    lr = min_lr + (max_lr - min_lr) * np.maximum(0, (1 - x))
    return lr


if __name__ == '__main__':
    x = []
    for i in range(100):
        t = get_triangler_lr(i, 10, 0.1, 1)
        x.append(t)

    show_graph(list(range(len(x))), x, "lr", "step", "lr vs step")
