__author__ = 'marvin'
import numpy as np


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def grad_softmax(softx):
    return softx * (1. - softx)


def relu(x):
    return np.where(x > 0., x, 0.)


def grad_relu(relux):
    return np.where(relux > 0., 1., 0.)


def sigmoid(x):
    return np.exp(x) / (np.exp(x) + 1.)


def grad_sigmoid(sigx):
    return sigx * (1. - sigx)


def identity(x):
    return x


def grad_identity(_):
    return 1.


def cross_entropy_loss(predictions, ytrue, epsilon=1e-8):
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    ce = -np.sum(ytrue*np.log(predictions), axis=-1)
    #print '\nce', predictions, ytrue, ce
    return ce


def grad_cross_entropy_loss(y, ytrue):
    return y - ytrue