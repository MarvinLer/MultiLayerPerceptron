import numpy as np

from src import functions
from src.layers import Dense, Dropout, Conv2D, Reshape
from src.utils import print_progress_bar
from examples.mlp_mnist.mnist_loader import plot_digit


class Sequential(object):
    def __init__(self, input_shape, cost_function='cross_entropy', learning_rate=1e-3, batch_size=32,
                 random_state=None):
        np.random.seed(random_state)
        self.input_shape = input_shape
        self.output_shape = input_shape
        self.n_layers = 0
        self.layers = []
        if cost_function == 'cross_entropy':
            self.loss, self.grad_loss = functions.cross_entropy_loss, functions.grad_cross_entropy_loss
        elif cost_function == 'squared_loss':
            self.loss, self.grad_loss = functions.squared_loss, functions.grad_cross_entropy_loss
        elif cost_function == 'gan_discriminator':
            self.loss, self.grad_loss = functions.discriminator_loss, functions.grad_discriminator_loss
        elif cost_function == 'gan_generator':
            self.loss, self.grad_loss = functions.generator_loss, functions.grad_generator_loss
        else:
            raise ValueError('%s cost function unknown' % cost_function)
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.last_output = None  # Container to save last network output for loss computation

    def add_dense_layer(self, n_neurons, activation='relu'):
        """
        Append a fully-connected layer to the end of the network.
        :param n_neurons: the number of output neurons desired
        :param activation: the type of activation function desired
        """
        layer = Dense(self.n_layers, self.output_shape, n_neurons, activation)
        self.layers.append(layer)
        self.n_layers += 1
        self.output_shape = layer.output_shape

    def add_conv2d_layer(self, filter_size, n_filters, stride=(1, 1), padding=None, activation='relu'):
        """
        Append a fully-connected layer to the end of the network.
        :param n_neurons: the number of output neurons desired
        :param activation: the type of activation function desired
        """
        layer = Conv2D(self.n_layers, self.output_shape, filter_size, n_filters, stride, padding, activation)
        self.layers.append(layer)
        self.n_layers += 1
        self.output_shape = layer.output_shape

    def add_dropout_layer(self, dropout=0.2):
        """
        Append a dropout layer to the end of the network.
        :param dropout: percetage of neurons to discard
        """
        self.layers.append(Dropout(id_layer=self.n_layers, input_shape=self.output_shape, dropout=dropout))
        self.n_layers += 1
        self.output_shape = self.output_shape

    def add_reshape_layer(self, newshape):
        """
        Append a dropout layer to the end of the network.
        :param dropout: percetage of neurons to discard
        """
        layer = Reshape(input_shape=self.output_shape, id_layer=self.n_layers, newshape=newshape)
        self.layers.append(layer)
        self.n_layers += 1
        self.output_shape = layer.output_shape

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
            print('\nStarting training')

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
                if verbose and i == n_steps - 1:
                    print_progress_bar(i + 1, n_steps,
                                       prefix='Epoch %02d/%02d step %04d/%04d' % (epoch, n_epochs, i + 1, n_steps),
                                       suffix='loss=%.4f acc=%.3f' % (np.mean(rolling_loss),
                                                                      np.mean(rolling_accuracy)))

            if verbose:
                val_mean, val_std, val_acc = self.get_metrics(xval, yval, verbose=False)
                print(' val_loss=%.4f val_acc=%.3f' % (val_mean, val_acc))

                #plot_digit(xval[0], self.forward(xval[0]), epoch)

    def predict(self, x, verbose=True):
        """
        Predict the given data using the network.
        :param x: array of data to predict
        :param verbose: True to print training stats
        """
        if verbose:
            print('\nStarting predictions')

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
            print('Error on predictions: %.5f +/- %.5f' % (np.mean(errors), np.std(errors)))
            print('Accuracy on predictions: %.4f' % accuracy)

        return np.mean(errors), np.std(errors), accuracy

    def summary(self):
        """
        Print network architecture on std.
        """
        n_columns = 4  # Layer name; input shape; output shape; number of parameters
        row_format = "{:>25}" * 4
        inter_line = "-" * 25 * n_columns
        print(inter_line)
        print(row_format.format("Layer", "Input shape", "Output shape", "N parameters"))
        print(inter_line)
        for layer in self.layers:
            name, input_shape, output_shape, n_parameters = layer.summary()
            print(row_format.format(name, input_shape, output_shape, n_parameters))
            print(inter_line)


class GAN(object):
    def __init__(self, n_inputs, n_latents, generator, discriminator, batch_size=10, learning_rate=1e-3,
                 prior_type='uniform'):
        self.generator = generator
        self.discriminator = discriminator
        self.n_inputs = n_inputs
        self.latent_dim = n_latents
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        # Determine the type of the prior ditribution
        if prior_type == 'uniform':
            self.prior = lambda size: np.random.uniform(low=-1., high=1., size=size)
        elif prior_type == 'normal':
            self.prior = lambda size: np.random.normal(loc=0., scale=1., size=size)
        else:
            raise ValueError('Unknow GAN prior distribution', prior_type)

        # Container to store intermediate results
        self.gz_gen = None
        self.gz_dis = None
        self.dgz_gen = None
        self.dgz_dis = None
        self.dx = None

    def forward(self, x, z_generator, z_discr, training=True):
        # Forward pass for generator with noise prior
        gz_gen = self.generator.forward(z_generator, training=training)
        self.gz_gen = gz_gen
        dgz_gen = self.discriminator.forward(gz_gen, training=training)
        self.dgz_gen = dgz_gen

        # Forward pass for discriminator with noise prior
        gz_dis = self.generator.forward(z_discr, training=training)
        self.gz_dis = gz_dis
        dgz_dis = self.discriminator.forward(gz_dis, training=training)
        self.dgz_dis = dgz_dis
        # Forward pass for discriminator with real prior
        dx = self.discriminator.forward(x, training=training)
        self.dx = dx

    def backpropagation(self, momentum):
        # Backpropagation for discriminator
        discriminator_loss = self.discriminator.loss(self.dx, self.dgz_dis)
        error = self.discriminator.grad_loss(self.dx, self.dgz_dis)
        for layer in self.discriminator.layers[::-1]:
            error = layer.backpropagation(error, self.learning_rate, momentum)

        # Backpropagation for generator
        generator_loss = self.generator.loss(self.dgz_gen)
        error = self.generator.grad_loss(error, self.dgz_gen)
        for layer in self.generator.layers[::-1]:
            error = layer.backpropagation(error, self.learning_rate, momentum)

        return discriminator_loss, generator_loss

    def fit(self, xtrain, n_epochs=None, n_steps=None, shuffle=True, momentum=None, verbose=True):
        if verbose:
            print('\nStarting training')

        if not n_epochs:
            n_epochs = 1
            n_steps //= self.batch_size
        else:
            n_steps = xtrain.shape[0] // self.batch_size

        for epoch in range(n_epochs):
            if shuffle:
                np.random.shuffle(xtrain)

            # Containers for stats printing
            rolling_discriminator_loss = []
            rolling_generator_loss = []
            for i in range(n_steps):
                # Select next batch
                x = xtrain[
                    i * self.batch_size % (xtrain.shape[0] + 1):(i + 1) * self.batch_size % (xtrain.shape[0] + 1)]
                # Draws random samples from normal distrib
                z_dis = np.random.rand(self.batch_size, self.latent_dim)
                z_gen = np.random.rand(self.batch_size, self.latent_dim)
                # Perform forward and backward passes
                self.forward(x, z_gen, z_dis, training=True)
                discriminator_loss, generator_loss = self.backpropagation(momentum=momentum)

                # Update printing stats and update prograss bar
                if verbose and i % max((n_steps // 50), 1) == 0:
                    rolling_discriminator_loss.append(discriminator_loss)
                    rolling_generator_loss.append(generator_loss)
                    print_progress_bar(i + 1, n_steps,
                                       prefix='Epoch %02d/%02d step %04d/%04d' % (epoch, n_epochs, i + 1, n_steps),
                                       suffix='gan loss=%.4f generator loss=%.4f discriminator loss=%.4f' % (
                                           np.mean(rolling_discriminator_loss[-n_steps // 50:]) + np.mean(
                                               rolling_generator_loss[-n_steps // 50:]),
                                           np.mean(rolling_discriminator_loss[-n_steps // 50:]),
                                           np.mean(rolling_generator_loss[-n_steps // 50:])))
                if verbose and i == n_steps - 1:
                    print_progress_bar(i + 1, n_steps,
                                       prefix='Epoch %02d/%02d step %04d/%04d' % (epoch, n_epochs, i + 1, n_steps),
                                       suffix='gan loss=%.4f generator loss=%.4f discriminator loss=%.4f' % (
                                           np.mean(rolling_discriminator_loss)+np.mean(rolling_generator_loss),
                                           np.mean(rolling_discriminator_loss),
                                           np.mean(rolling_generator_loss)))
                if i % 1000 == 0:
                    plot_digit(None, self.gz_gen[0], epoch * n_steps + i)

            print(' ')

    def summary(self):
        print('GENERATOR')
        self.generator.summary()
        print('DISCRIMINATOR')
        self.discriminator.summary()
