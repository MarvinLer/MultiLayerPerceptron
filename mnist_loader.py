import os
import struct
import numpy as np

n_classes = 10


def preprocess(images, labels, one_hot, normalize):
    # One-hot labels
    if one_hot:
        tmp = np.zeros((len(labels), n_classes))
        tmp[:, labels] = 1.
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
