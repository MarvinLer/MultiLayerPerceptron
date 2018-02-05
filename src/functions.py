__author__ = 'marvin'
import numpy as np
np.seterr(all='raise')


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=-1)[:, np.newaxis]


def grad_softmax(softx):
    return softx * (1. - softx)


def relu(x):
    return np.where(x > 0., x, 0.)


def grad_relu(relux):
    return np.where(relux > 0., 1., 0.)


def sigmoid(x):
    return 1. / (np.exp(-x) + 1.)


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


def squared_loss(y, ytrue):
    return (y - ytrue) ** 2.


def grad_squared_loss(y, ytrue):
    return 2. * (y - ytrue)


def dropout(x, perc_dropout):
    keep = np.random.choice([0, 1], size=x.shape, p=[perc_dropout, 1.-perc_dropout])
    return np.where(keep, x, 0.), keep


def grad_dropout(keep_matrix):
    return np.where(keep_matrix, 1., 0.)


def log_loss(x):
    return np.log10(x)


def grad_log_loss(y, _):
    return 1. / y


def discriminator_loss(dx, dgz):
    return np.log10(dx) + np.log10(1. - dgz)


def grad_discriminator_loss(dx, dgz):
    return -1. / dx + 1. / (1. - dgz)


def generator_loss(dgz):
    return -np.log10(dgz)


def grad_generator_loss(grad_dgz, dgz):
    return -grad_dgz / dgz
