import matplotlib.pyplot as plt
import numpy as np

def plot_mnist(x_tensor):

    x = x_tensor.detach().numpy()
    plt.imshow(x.reshape(28,28),cmap="binary_r")
    plt.show()