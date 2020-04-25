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