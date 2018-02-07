import numpy as np
from src import functions

__author__ = 'marvin'


class Dense(object):
    def __init__(self, id_layer, input_shape, n_neurons, activation='relu'):
        self.inference = True

        self.id_layer = id_layer
        self.name = 'Dense layer %d' % self.id_layer
        self.n_neurons = n_neurons  # Output number of neurons
        self.output_shape = (n_neurons,)
        self.input_shape = (input_shape,) if isinstance(input_shape, int) else input_shape  # Input number of neurons
        # Activation function with its gradient
        if activation == 'relu':
            self.activation = functions.relu
            self.grad_activation = functions.grad_relu
        elif activation == 'softmax':
            self.activation = functions.softmax
            self.grad_activation = functions.grad_softmax
        elif activation == 'sigmoid':
            self.activation = functions.sigmoid
            self.grad_activation = functions.grad_sigmoid
        elif activation is None:
            self.activation = functions.identity
            self.grad_activation = functions.grad_identity
        else:
            raise ValueError('select activation among "relu", "softmax" or "sigmoid"')
        # Parameters
        self.w = self.init_weights()
        self.b = self.init_biases()
        # Update history
        self.dw = np.zeros(self.w.shape)
        self.db = np.zeros(self.b.shape)

        # Saved computed variables to accelerate backpropagation computation
        self.x = np.zeros(input_shape)  # Layer inputs
        self.z = np.zeros(n_neurons)  # Layer linear operation
        self.a = np.zeros(n_neurons)  # Layer activations

    def init_biases(self):
        """
        Initializes layer biases (one per output neuron)
        :return: a np array with bias init values
        """
        # Biases put to 0. at beginning
        return np.zeros(self.n_neurons)

    def init_weights(self):
        """
        Initializes layer weights (one per output neuron)
        :return: a np array with bias init values
        """
        # Xavier initialization
        var = 2. / (float(self.n_neurons + np.prod(self.input_shape)))
        return np.random.normal(loc=0., scale=np.sqrt(var), size=(self.input_shape + (self.n_neurons,)))

    def forward(self, x):
        """
        Performs forward pass of the layer using the input x.
        :param x: input array
        """
        # Forward pass
        self.x = x
        #self.z = self.w.T.dot(self.x) + self.b
        self.z = self.x.dot(self.w) + self.b
        self.a = self.activation(self.z)
        return self.a

    def backpropagation(self, da, lr, momentum):
        """
        Performs backpropagation from activation to input layer using chain rule.
        :param da: backpropagation input error
        :param lr: learning rate (step for the gradient descent)
        :return: computed loss from forward pass and true labels
        """
        # backprop self.a = self.activation(self.z)
        dz = self.grad_activation(self.a) * da  # Dot-product
        # backprop on w for self.z = self.x.dot(self.w) + self.b
        dw = np.matmul(self.x[:, :, np.newaxis], dz[:, np.newaxis, :])
        # backprop on b for self.z = self.x.dot(self.w) + self.b
        db = 1. * dz
        # backprop self.x = x
        dx = dz.dot(self.w.T)

        # Compute the vectors of parameters update
        db = np.mean(db, axis=0)
        dw = np.mean(dw, axis=0)
        # Momentum
        if momentum:
            self.db = momentum * self.db + lr * db
            self.dw = momentum * self.dw + lr * dw
        else:
            self.db = lr * db
            self.dw = lr * dw

        assert dw.shape == self.w.shape
        assert db.shape == self.b.shape

        self.b -= self.db
        self.w -= self.dw
        return dx  # Propagates error signal

    def summary(self):
        return self.name, self.input_shape, self.output_shape, self.w.size + self.b.size


class Conv2D(object):
    def __init__(self, id_layer, input_shape, filter_size, n_filters, stride, padding, activation='relu'):
        self.inference = True

        self.id_layer = id_layer
        self.name = 'Convolutional layer %d' % self.id_layer
        self.input_shape = input_shape  # Input shape (channels last)
        self.n_filters = n_filters  # Number of filters
        # Size of filters
        if isinstance(filter_size, int):
            self.filter_size = (filter_size, filter_size)  # Size of filters
        elif isinstance(filter_size, tuple) and len(filter_size) == 2 and False not in [isinstance(f, int) for f in
                                                                                        filter_size]:
            self.filter_size = filter_size
        else:
            raise ValueError('Conv2D: filter_size should be a 2-size list of int or an int')
        self.n_neurons = reduce(lambda x, y: x * y, self.filter_size) * self.n_filters * self.input_shape[-1]

        self.stride = stride
        self.padding = (0, 0) if not padding else padding
        self.output_shape = tuple(map(lambda (w, f, p, s): (w - f + 2 * p) // s + 1,
                                      zip(self.input_shape, self.filter_size, self.padding, self.stride))) + (self.n_filters,)

        # Activation function with its gradient
        if activation == 'relu':
            self.activation = functions.relu
            self.grad_activation = functions.grad_relu
        elif activation == 'softmax':
            self.activation = functions.softmax
            self.grad_activation = functions.grad_softmax
        elif activation == 'sigmoid':
            self.activation = functions.sigmoid
            self.grad_activation = functions.grad_sigmoid
        elif activation is None:
            self.activation = functions.identity
            self.grad_activation = functions.grad_identity
        else:
            raise ValueError('select activation among "relu", "softmax" or "sigmoid"')
        # Parameters
        self.w = self.init_weights()
        self.b = self.init_biases()
        # Update history
        self.dw = np.zeros(self.w.shape)
        self.db = np.zeros(self.b.shape)

        # Saved computed variables to accelerate backpropagation computation
        self.x = None  # Layer inputs
        self.x_windowed = None  # Reformatted layer inputs
        self.inner_z = None  # Preformatted linear activation
        self.z = None  # Layer linear operation
        self.a = None  # Layer activations

    def init_biases(self):
        """
        Initializes layer biases (one per output neuron)
        :return: a np array with bias init values
        """
        # Biases put to 0. at beginning
        return np.zeros(self.n_filters)

    def init_weights(self):
        """
        Initializes layer filters i.e. a list of self.n_filters elements of shape (filter_size_x, filter_size_y,
        channels).
        :return: a np array with bias init values
        """
        # Xavier initialization
        var = 1. / float(self.n_neurons)
        return np.random.normal(loc=0., scale=np.sqrt(var),
                                size=((self.n_filters,) + self.filter_size + (self.input_shape[-1],)))

    def forward(self, x):
        """
        Performs forward pass of the layer using the input x.
        :param x: input array
        """
        # Forward pass
        self.x = x
        # Compute the grid of windows considered in the input using stride, filter size and padding as a list of
        # tuples of 2 elements
        windows_idx = [(slice(i, i + self.filter_size[0], 1), slice(j, j + self.filter_size[1], 1)) for i in
                       range(self.output_shape[0]) for j in range(self.output_shape[1])]
        print 'x', x.shape
        self.x_windowed = np.apply_along_axis(lambda xi: [xi[wx, wy, :] for wx, wy in windows_idx], axis=0, arr=x)
        for window in self.x_windowed:
            assert window.shape[1, 2] == (self.filter_size[0], self.filter_size[1])
        # For each batch element, for each windowed input, compute convolution with bias
        self.inner_z = np.apply_along_axis(lambda window:
                                           np.apply_along_axis(lambda f, b: np.sum(f * window) + b, axis=0,
                                                               arr=np.array(list(zip(self.w, self.b)))),
                                           axis=1, arr=self.x_windowed)
        assert self.inner_z.shape == (self.x.shape[0], self.output_shape[0] * self.output_shape[1], self.n_filters)
        self.z = self.inner_z.reshape((self.x.shape[0], self.output_shape[0], self.output_shape[1], self.n_filters))
        self.a = self.activation(self.z)
        return self.a

    def backpropagation(self, da, lr, momentum):
        """
        Performs backpropagation from activation to input layer using chain rule.
        :param da: backpropagation input error
        :param lr: learning rate (step for the gradient descent)
        :return: computed loss from forward pass and true labels
        """
        dz = self.grad_activation(self.a) * da  # Dot-product
        dz_reformatted = dz.reshape((self.x.shape[0], self.output_shape[0] * self.output_shape[1], self.n_filters))

        dw = np.sum(np.matmul(dz_reformatted[:, :, :, np.newaxis], self.x_windowed[:, :, np.newaxis, :, :]), axis=1)

        db = np.sum(dz_reformatted, axis=1)
        # backprop self.x = x
        dx = dz.dot(self.w.T)

        # Compute the vectors of parameters update by averaging the batch updates
        db = np.mean(db, axis=0)
        dw = np.mean(dw, axis=0)
        # Momentum
        if momentum:
            self.db = momentum * self.db + lr * db
            self.dw = momentum * self.dw + lr * dw
        else:
            self.db = lr * db
            self.dw = lr * dw

        assert dw.shape == self.w.shape
        assert db.shape == self.b.shape

        self.b -= self.db
        self.w -= self.dw
        return dx  # Propagates error signal

    def summary(self):
        return self.name, self.input_shape, self.output_shape, self.w.size + self.b.size


class Dropout(object):
    def __init__(self, id_layer, input_shape, dropout):
        self.inference = False

        self.id_layer = id_layer
        self.name = 'Dropout layer %d' % self.id_layer
        self.input_shape = input_shape
        self.output_shape = input_shape
        self.dropout = dropout  # Percentage of elements to discard
        # Activation function with its gradient
        self.activation = functions.dropout
        self.grad_activation = functions.grad_dropout

        # Saved computed variables to accelerate backpropagation computation
        self.x = np.zeros(input_shape)  # Layer inputs
        self.a = np.zeros(input_shape)  # Layer activations
        self.keep_matrix = np.zeros(input_shape)  # Keep probability matrix to easily compute layer backprop

    def forward(self, x):
        """
        Performs forward pass of the layer using the input x.
        :param x: input array
        """
        # Forward pass
        self.x = x
        self.a, self.keep_matrix = self.activation(self.x, self.dropout)
        return self.a

    def backpropagation(self, da, *args):
        """
        Performs backpropagation from activation to input layer using chain rule: propagate gradient neuron by neuron
        if neuron was active during forward pass.
        :param da: backpropagation input error
        :param _: learning rate (step for the gradient descent); not used
        :return: computed loss from forward pass and true labels
        """
        # backprop self.a = self.activation(self.x)
        dx = self.grad_activation(self.keep_matrix) * da
        return dx  # Propagates error signal

    def summary(self):
        return self.name, self.input_shape, self.output_shape, None


class Reshape(object):
    def __init__(self, id_layer, input_shape, newshape):
        self.inference = True

        self.id_layer = id_layer
        self.name = 'Reshape layer %d' % self.id_layer
        self.oldshape = input_shape
        self.input_shape = input_shape
        newshape = np.asarray(newshape)
        if -1 in newshape:
            assert np.sum(newshape == -1) == 1, 'Cannot put several -1 in Reshape layer'
            if np.prod(input_shape) % np.prod(newshape):
                raise ValueError(
                    'Cannot reshape input of shape (%s) into new shape (%s)' %
                    (', '.join(map(str, input_shape)), ', '.join(map(str, newshape))))
            if np.sum(newshape == -1) == len(newshape):
                newshape[newshape == -1] = np.prod(input_shape)
            else:
                newshape[newshape == -1] = np.prod(input_shape) / np.prod(newshape)
        self.newshape = tuple(newshape)
        self.output_shape = self.newshape

        # Saved computed variables to accelerate backpropagation computation
        self.x = np.zeros(input_shape)  # Layer inputs
        self.a = np.zeros(self.newshape)  # Layer activations

    def forward(self, x):
        """
        Performs forward pass of the layer using the input x.
        :param x: input array
        """
        # Forward pass
        self.x = x
        self.a = np.reshape(x, self.newshape)
        return self.a

    def backpropagation(self, da, *args):
        """
        Performs backpropagation from activation to input layer using chain rule: propagate gradient neuron by neuron
        if neuron was active during forward pass.
        :param da: backpropagation input error
        :param _: learning rate (step for the gradient descent); not used
        :return: computed loss from forward pass and true labels
        """
        # backprop self.a = self.activation(self.x)
        dx = np.reshape(da, self.oldshape)
        return dx  # Propagates error signal

    def summary(self):
        return self.name, self.input_shape, self.output_shape, None
