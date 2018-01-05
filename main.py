__author__ = 'marvin'
from mnist_loader import load_mnist
from multilayerperceptron import MLP

train = load_mnist('mnist', 'train')
test = load_mnist('mnist', 't10k')