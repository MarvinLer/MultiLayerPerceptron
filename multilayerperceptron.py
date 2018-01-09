import numpy as np
from utils import print_progress_bar


class DenseLayer(object):
    def __init__(self, n_neurons, n_inputs, id_layer, activation='relu'):
        self.id_layer = id_layer
        self.name = 'Dense layer %d' % self.id_layer
        self.n_neurons = n_neurons  # Output number of neurons
        self.n_inputs = n_inputs  # Input number of neurons
        # Activation function with its gradient
        if activation == 'relu':
            self.activation = relu
            self.grad_activation = grad_relu
        elif activation == 'softmax':
            self.activation = softmax
            self.grad_activation = grad_softmax
        elif not activation:
            self.activation = identity
            self.grad_activation = grad_identity
        else:
            raise ValueError('select activation among "relu" or "softmax"')
        # Parameters
        self.w = self.init_weights()
        self.b = self.init_biases()

        # Saved computed variables to accelerate backpropagation computation
        self.x = np.zeros(n_inputs)  # Layer inputs
        self.z = np.zeros(n_neurons)  # Layer linear operation
        self.a = np.zeros(n_neurons)  # Layer activations
        self.params_a = np.zeros(n_neurons)  # Contains block computation of activation

    def init_biases(self):
        # Biases put to 0. at beginning
        return np.zeros(self.n_neurons)

    def init_weights(self):
        # Xavier initialization
        var = 2. / (float(self.n_neurons + self.n_inputs))
        return np.random.normal(loc=0., scale=np.sqrt(var), size=(self.n_inputs, self.n_neurons))

    def forward(self, x):
        # Forward pass
        self.x = x
        self.z = self.w.T.dot(self.x) + self.b
        self.params_a, self.a = self.activation(self.z)
        return self.a

    def backpropagation(self, da, lr):
        # backprop self.a = self.activation(self.z)
        dz = self.grad_activation(self.z, *self.params_a) * da  # Dot-product
        assert dz.shape == self.a.shape, 'Expected shape ' + self.a.shape + ' got shape ' + dz.shape
        # backprop on w for self.z = self.w.T * self.x + self.b
        dw = np.outer(self.x, dz)
        assert dw.shape == self.w.shape, 'Expected shape ' + self.w.shape + ' got shape ' + dw.shape
        # backprop on b for self.z = self.w.T * self.x + self.b
        db = 1. * dz
        # backprop self.x = x
        dx = self.w.dot(dz)

        self.b -= lr * db
        self.w -= lr * dw
        return dx  # Propagates error signal


class MLP(object):
    def __init__(self, n_inputs, learning_rate=1e-3, cost_function='cross_entropy', random_state=None):
        np.random.seed(random_state)
        self.n_inputs = n_inputs
        self.n_outputs = n_inputs
        self.n_layers = 0
        self.layers = []
        if cost_function == 'cross_entropy':
            self.loss, self.grad_loss = softmax_cross_entropy_loss, grad_softmax_cross_entropy_loss
        self.learning_rate = learning_rate

        self.last_output = None  # Save last network output

    def add_layer(self, n_neurons, activation='relu'):
        self.n_layers += 1
        self.layers.append(DenseLayer(n_neurons, self.n_outputs, self.n_layers, activation))
        self.n_outputs = n_neurons

    def summary(self):
        """
        Print network architecture on std
        """
        n_columns = 4  # Layer name; input shape; output shape; number of parameters
        row_format = "{:>15}" * 4
        inter_line = "*" * 15 * n_columns
        print inter_line
        print row_format.format("Layer", "Input shape", "Output shape", "N parameters")
        print inter_line
        for layer in self.layers:
            print row_format.format(layer.name, layer.x.shape, layer.a.shape, layer.w.size+layer.b.size)
            print inter_line

    def forward(self, x):
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        self.last_output = output

    def backpropagation(self, ytrue):
        y, loss = self.loss(self.last_output, ytrue)
        error = self.grad_loss(y, ytrue)
        for layer in self.layers[::-1]:
            error = layer.backpropagation(error, self.learning_rate)
        return loss

    def fit(self, xtrain, ytrain, n_steps, n_print=1):
        print '\nStarting training'
        print_progress_bar(0, n_steps, prefix='Step %d/%d' % (0, n_steps))
        for i in range(n_steps):
            self.forward(xtrain[i])
            loss = self.backpropagation(ytrain[i])
            if i % n_print == 0:
                print_progress_bar(i, n_steps, prefix='Step %d/%d' % (i, n_steps), suffix='loss=%.4f' % loss)
        print_progress_bar(n_steps, n_steps, prefix='Step %d/%d' % (n_steps, n_steps), suffix='loss=%.4f' % loss)

    @staticmethod
    def log_likelihood_loss(a, y):
        return -np.dot(y, a.softmax(a).transpose())


def softmax(x):
    # Computation: decompose formula into elementary segments to easily compute backprop afterwards
    num = np.exp(x)
    den = np.sum(num, axis=0)
    invden = 1. / den
    softx = num * invden
    return [num, den, invden], softx


def grad_softmax(dy, num, den, invden):
    # backprop softx = num * invden
    dnum = invden
    dinvden = num
    # backprop invden = 1.0 / den
    dden = (-1. / (den ** 2)) * dinvden
    # backprop den = np.sum(num, axis=0)
    dnum += 1. * dden
    # backprop num = np.exp(x)
    dx = num * dnum

    return dy * dx


def softmax_cross_entropy_loss(z, ytrue):
    _, y = softmax(z)
    return y, -np.sum(ytrue * np.log(y), axis=-1)


def grad_softmax_cross_entropy_loss(y, ytrue):
    return ytrue - y


def relu(x):
    relux = np.where(x > 0., x, 0.)
    return [relux], relux


def grad_relu(dy, relux):
    dx = np.where(relux > 0., 1., 0.)
    return dy * dx


def identity(x):
    return [x], x


def grad_identity(dy, x):
    return dy * 1.


