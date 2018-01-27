import numpy as np
from functions import *
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
        elif activation == 'sigmoid':
            self.activation = sigmoid
            self.grad_activation = grad_sigmoid
        elif activation is None:
            self.activation = identity
            self.grad_activation = grad_identity
        else:
            raise ValueError('select activation among "relu", "softmax" or "sigmoid"')
        # Parameters
        self.w = self.init_weights()
        self.b = self.init_biases()

        # Saved computed variables to accelerate backpropagation computation
        self.x = np.zeros(n_inputs)  # Layer inputs
        self.z = np.zeros(n_neurons)  # Layer linear operation
        self.a = np.zeros(n_neurons)  # Layer activations

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
        self.a = self.activation(self.z)
        return self.a

    def backpropagation(self, da, lr):
        # backprop self.a = self.activation(self.z)
        dz = self.grad_activation(self.a) * da  # Dot-product
        # backprop on w for self.z = self.w.T * self.x + self.b
        dw = np.outer(self.x, dz)
        # backprop on b for self.z = self.w.T * self.x + self.b
        db = 1. * dz
        # backprop self.x = x
        #dx = self.w.dot(dz)
        dx = dz.dot(self.w.T)

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
            self.loss, self.grad_loss = cross_entropy_loss, grad_cross_entropy_loss
        self.learning_rate = learning_rate

        self.last_output = None  # Save last network output

    def add_layer(self, n_neurons, activation='relu'):
        """
        Append a layer to the end of the network.
        :param n_neurons: the number of output neurons desired
        :param activation: the type of activation function desired
        """
        self.layers.append(DenseLayer(n_neurons, self.n_outputs, self.n_layers, activation))
        self.n_layers += 1
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
            print row_format.format(layer.name, layer.x.shape, layer.a.shape, layer.w.size + layer.b.size)
            print inter_line

    def forward(self, x):
        """
        Performs forward pass using the input x.
        :param x: input array
        """
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        self.last_output = output
        return self.last_output

    def backpropagation(self, ytrue):
        """
        Performs backpropagation throughout the whole network using chain rule.
        :param ytrue: true array of labels from the previous input
        :return: computed loss from forward pass and true labels
        """
        loss = self.loss(self.last_output, ytrue)
        error = self.grad_loss(self.last_output, ytrue)
        for layer in self.layers[::-1]:
            error = layer.backpropagation(error, self.learning_rate)
        return loss

    def fit(self, xtrain, ytrain, n_steps, verbose=True):
        """
        Train the network.
        :param xtrain: array containing the training data
        :param ytrain: array containing the training labels
        :param n_steps: number of steps for training
        :param verbose: True to print training stats
        """
        if verbose:
            print '\nStarting training'

        for i in range(n_steps):
            self.forward(xtrain[i % len(xtrain)])
            loss = self.backpropagation(ytrain[i % len(xtrain)])
            if verbose:
                print_progress_bar(i + 1, n_steps, prefix='Step %d/%d' % (i + 1, n_steps), suffix='loss=%.4f' % loss)

    def predict(self, x, verbose=True):
        """
        Predict the given data using the network.
        :param x: array of data to predict
        :param verbose: True to print training stats
        """
        if verbose:
            print '\nStarting predictions'

        predictions = np.zeros((len(x), self.n_outputs))
        n_steps = len(x)
        for i, data in enumerate(x):
            predictions[i] = self.forward(data)
            if verbose:
                print_progress_bar(i + 1, n_steps, prefix='Step %d/%d' % (i + 1, n_steps))

        return predictions

    def get_metrics(self, x, y, verbose=True):
        predictions = self.predict(x, verbose=verbose)
        # Compute loss metrics
        errors = self.loss(predictions, y)
        print 'Error on predictions: %.5f +/- %.5f' % (np.mean(errors), np.std(errors))
        # Compute accuracy metrics
        accuracy = np.mean(np.argmax(predictions, axis=-1) == np.argmax(y, axis=-1))
        print 'Accuracy on predictions: %.4f' % accuracy
        import pylab as plt
        plt.hist(errors, bins=30)
        plt.show()
        return