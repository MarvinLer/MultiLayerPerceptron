import os
import struct
import numpy as np
import matplotlib.pyplot as plt

n_classes = 10


def preprocess(images, labels, one_hot, normalize):
    # One-hot labels
    if one_hot:
        tmp = np.zeros((len(labels), n_classes))
        for i, l in enumerate(labels):
            tmp[i, l] = 1.
        labels = tmp

    # Normalize pixel input to [0, 1]
    if normalize:
        images = images.astype(np.float32)
        images /= 255.

    return images, labels


def load_mnist(path, kind='train', one_hot=True, normalize=True):
    labels_path = os.path.join('data', path, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join('data', path, '%s-images-idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lpath:
        _, _ = struct.unpack('>II', lpath.read(8))
        labels = np.fromfile(lpath, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        _, _, _, _ = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    images, labels = preprocess(images, labels, one_hot, normalize)

    return images, labels


def plot_digit(init, digit, i):
    pixels = np.array(digit * 255., dtype='uint8')
    if init is not None:
        pixels_init = np.array(init * 255., dtype='uint8')

    # Reshape the array into 28 x 28 array (2-dimensional array)
    pixels = pixels.reshape((28, 28))
    if init is not None:
        pixels_init = pixels_init.reshape((28, 28))

    # Plot
    plt.imshow(pixels, cmap='gray')
    plt.savefig('%d.png' % i)
    if init is not None:
        plt.imshow(pixels_init, cmap='gray')
        plt.savefig('%d_init.png' % i)
    return
