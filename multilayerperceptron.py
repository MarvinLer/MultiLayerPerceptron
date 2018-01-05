import numpy as np
from tqdm import tqdm


def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x), axis=0)
    return f_x, f_x * (1. - f_x)  # Function and its gradient


def relu(x):
    f_x = np.max([np.zeros(len(x)), x], axis=0)
    return f_x, np.where(f_x > 0., np.ones(len(x)), np.zeros(len(x)))


class DenseLayer(object):
    def __init__(self, n_neurons, n_inputs, activation='relu'):
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs
        if activation == 'relu':
            self.activation, self.grad_activation = relu
        elif activation == 'softmax':
            self.activation, self.grad_activation = softmax
        else:
            raise ValueError('select activation among "relu" or "softmax"')
        self.w = self.init_weights()
        self.b = self.init_biases()

        self.last_inputs = np.zeros(n_inputs)
        self.last_preactivations = np.zeros(n_neurons)
        self.last_activations = np.zeros(n_neurons)

        self.is_last_layer = True

    def init_biases(self):
        return np.zeros(self.n_neurons)

    def init_weights(self):
        # Xavier initialization
        var = 2. / (float(self.n_neurons+self.n_inputs))
        return np.random.normal(loc=0., scale=np.sqrt(var), size=(self.n_inputs, self.n_neurons))

    def set_is_not_last_layer(self):
        self.is_last_layer = False

    def forward(self, x):
        self.last_inputs = x
        self.last_preactivations = self.w.T * x + self.b
        self.last_activations = self.activation(self.last_preactivations)
        return self.last_activations

    def backward(self, e, lr):
        e = np.dot(e, self.grad_activation(self.last_preactivations))
        self.b -= lr * e
        self.w -= lr * e * self.last_inputs
        return self.w * e  # Propagates error signal


class MLP(object):
    def __init__(self, input_shape, n_classes, random_state=None):
        self.input_shape = input_shape
        self.output_shape = n_classes

    @staticmethod
    def cross_entropy_loss(a, y):
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def log_likelihood_loss(a, y):
        return -np.dot(y, a.softmax(a).transpose())
