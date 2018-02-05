import numpy as np

from src import functions
from src.utils import print_progress_bar


class DenseLayer(object):
    def __init__(self, n_neurons, n_inputs, id_layer, activation='relu'):
        self.inference = True

        self.id_layer = id_layer
        self.name = 'Dense layer %d' % self.id_layer
        self.n_neurons = n_neurons  # Output number of neurons
        self.n_inputs = n_inputs  # Input number of neurons
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
        self.x = np.zeros(n_inputs)  # Layer inputs
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
        var = 2. / (float(self.n_neurons + self.n_inputs))
        return np.random.normal(loc=0., scale=np.sqrt(var), size=(self.n_inputs, self.n_neurons))

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
        assert dw.shape == (self.x.shape[0], self.n_inputs, self.n_neurons), dw.shape
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

        self.b -= self.db
        self.w -= self.dw
        return dx  # Propagates error signal


class DropoutLayer(object):
    def __init__(self, n_neurons, id_layer, dropout):
        self.inference = False

        self.id_layer = id_layer
        self.name = 'Dropout layer %d' % self.id_layer
        self.dropout = dropout  # Percentage of elements to discard
        # Activation function with its gradient
        self.activation = functions.dropout
        self.grad_activation = functions.grad_dropout

        # Saved computed variables to accelerate backpropagation computation
        self.x = np.zeros(n_neurons)  # Layer inputs
        self.a = np.zeros(n_neurons)  # Layer activations
        self.keep_matrix = np.zeros(n_neurons)  # Keep probability matrix to easily compute layer backprop

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


class MLP(object):
    def __init__(self, n_inputs, cost_function='cross_entropy', learning_rate=1e-3, batch_size=32, random_state=None):
        np.random.seed(random_state)
        self.n_inputs = n_inputs
        self.n_outputs = n_inputs
        self.n_layers = 0
        self.layers = []
        if cost_function == 'cross_entropy':
            self.loss, self.grad_loss = functions.cross_entropy_loss, functions.grad_cross_entropy_loss
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.last_output = None  # Container to save last network output for loss computation

    def add_dense_layer(self, n_neurons, activation='relu'):
        """
        Append a fully-connected layer to the end of the network.
        :param n_neurons: the number of output neurons desired
        :param activation: the type of activation function desired
        """
        self.layers.append(DenseLayer(n_neurons, self.n_outputs, self.n_layers, activation))
        self.n_layers += 1
        self.n_outputs = n_neurons

    def add_dropout_layer(self, dropout=0.2):
        """
        Append a dropout layer to the end of the network.
        :param n_neurons: the number of output neurons desired
        :param dropout: percetage of neurons to discard
        """
        self.layers.append(DropoutLayer(n_neurons=self.n_outputs, id_layer=self.n_layers, dropout=dropout))
        self.n_layers += 1
        self.n_outputs = self.n_outputs

    def forward(self, x, training=True):
        """
        Performs forward pass of the whole network using the input x.
        :param x: input array
        :param training: True if the network is in training phase (then forward pass every layer) else Flase for
        inference phase
        """
        output = x
        for layer in self.layers:
            # If network is in inference mode, donc forward pass the layers that don't act at inference (e.g. dropout)
            if not training and not layer.inference:
                continue
            output = layer.forward(output)
        self.last_output = output
        return self.last_output

    def backpropagation(self, ytrue, momentum):
        """
        Performs backpropagation throughout the whole network using chain rule.
        :param ytrue: true array of labels from the previous input
        :return: computed loss from forward pass and true labels
        """
        loss = self.loss(self.last_output, ytrue)
        error = self.grad_loss(self.last_output, ytrue)
        for layer in self.layers[::-1]:
            error = layer.backpropagation(error, self.learning_rate, momentum)
        return loss

    @staticmethod
    def _shuffle_data(xtrain, ytrain):
        perm = np.random.permutation(range(xtrain.shape[0]))
        return xtrain[perm], ytrain[perm]

    def fit(self, xtrain, ytrain, xval, yval, n_epochs=None, n_steps=None, shuffle=True, momentum=None, verbose=True):
        """
        Train the network.
        :param xtrain: array containing the training data
        :param ytrain: array containing the training labels
        :param n_epochs: number of desired training epochs or None
        :param n_steps: number of steps for training; used only when n_epochs is None
        :param verbose: True to print training stats
        :param shuffle: True to shuffle data before each epoch; shuffled only at beginning if n_epochs equal to None
        :param xval: validation data
        :param yval: validation labels
        :param momentum: float between 0. and 1. for gradient descent momentum; None for no momentum
        """
        if verbose:
            print '\nStarting training'

        if not n_epochs:
            n_epochs = 1
            n_steps //= self.batch_size
        else:
            n_steps = xtrain.shape[0] // self.batch_size

        for epoch in range(n_epochs):
            if shuffle:
                xtrain, ytrain = self._shuffle_data(xtrain, ytrain)

            # Containers for stats printing
            rolling_loss = []
            rolling_accuracy = []
            for i in range(n_steps):
                # Select next batch
                x = xtrain[
                    i * self.batch_size % (xtrain.shape[0] + 1):(i + 1) * self.batch_size % (xtrain.shape[0] + 1)]
                y = ytrain[
                    i * self.batch_size % (xtrain.shape[0] + 1):(i + 1) * self.batch_size % (xtrain.shape[0] + 1)]
                # Perform forward and backward passes
                self.forward(x, training=True)
                loss = self.backpropagation(y, momentum=momentum)

                # Update printing stats and update prograss bar
                if verbose and i % max((n_steps // 20), 1) == 0:
                    rolling_loss.append(loss)
                    rolling_accuracy.append(np.argmax(self.last_output, axis=-1) == np.argmax(y, axis=-1))
                    print_progress_bar(i + 1, n_steps,
                                       prefix='Epoch %02d/%02d step %04d/%04d' % (epoch, n_epochs, i + 1, n_steps),
                                       suffix='loss=%.4f acc=%.3f' % (np.mean(rolling_loss[-n_steps // 50:]),
                                                                      np.mean(rolling_accuracy[-n_steps // 50:])))
                if verbose and i == n_steps-1:
                    print_progress_bar(i + 1, n_steps,
                                       prefix='Epoch %02d/%02d step %04d/%04d' % (epoch, n_epochs, i + 1, n_steps),
                                       suffix='loss=%.4f acc=%.3f' % (np.mean(rolling_loss),
                                                                      np.mean(rolling_accuracy)))

            if verbose:
                val_mean, val_std, val_acc = self.get_metrics(xval, yval, verbose=False)
                print ' val_loss=%.4f val_acc=%.3f' % (val_mean, val_acc)

    def predict(self, x, verbose=True):
        """
        Predict the given data using the network.
        :param x: array of data to predict
        :param verbose: True to print training stats
        """
        if verbose:
            print '\nStarting predictions'

        predictions = []
        n_steps = len(x) // self.batch_size if not len(x) % self.batch_size else len(x) // self.batch_size + 1
        for batch in range(n_steps):
            predictions.extend(self.forward(x[batch * self.batch_size:min(x.shape[0], (batch + 1) * self.batch_size)],
                                            training=False))
            if verbose:
                print_progress_bar(batch + 1, n_steps, prefix='Step %d/%d' % (batch + 1, n_steps))

        return np.asarray(predictions)

    def get_metrics(self, x, y, verbose=True):
        """
        Print computed metrics on the dataset (x, y).
        :param x: input data to predict
        :param y: labels associated to x
        :param verbose: True to print prediction advancement
        """
        predictions = self.predict(x, verbose=verbose)
        #print predictions[:10]
        # Compute loss metrics
        errors = self.loss(predictions, y)
        # Compute accuracy metrics
        accuracy = np.mean(np.argmax(predictions, axis=-1) == np.argmax(y, axis=-1))
        if verbose:
            print 'Error on predictions: %.5f +/- %.5f' % (np.mean(errors), np.std(errors))
            print 'Accuracy on predictions: %.4f' % accuracy
        #import pylab as plt
        #plt.hist(errors, bins=30)
        #plt.show()

        return np.mean(errors), np.std(errors), accuracy

    def summary(self):
        """
        Print network architecture on std.
        """
        n_columns = 4  # Layer name; input shape; output shape; number of parameters
        row_format = "{:>15}" * 4
        inter_line = "*" * 15 * n_columns
        print inter_line
        print row_format.format("Layer", "Input shape", "Output shape", "N parameters")
        print inter_line
        for layer in self.layers:
            print row_format.format(layer.name, layer.x.shape, layer.a.shape,
                                    layer.w.size + layer.b.size if hasattr(layer, 'w') else None)
            print inter_line
