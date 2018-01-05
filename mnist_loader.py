import os
import struct
import numpy as np


def load_mnist(path, kind='train'):
    """  load_mnist('mnist') => returns (images, labels)"""
    labels_path = os.path.join('data', path, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join('data', path, '%s-images-idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lpath:
        _, _ = struct.unpack('>II', lpath.read(8))
        labels = np.fromfile(lpath, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        _, _, _, _ = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    return images, labels
