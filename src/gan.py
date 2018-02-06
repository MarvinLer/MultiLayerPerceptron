import numpy as np
from src.models import Sequential
from src.utils import print_progress_bar
from examples.mlp_mnist.mnist_loader import plot_digit


class GAN(object):
    def __init__(self, n_inputs, n_latents, generator, discriminator, batch_size=10, learning_rate=1e-3):
        self.generator = generator
        self.discriminator = discriminator
        self.n_inputs = n_inputs
        self.latent_dim = n_latents
        self.batch_size = batch_size

        self.learning_rate = learning_rate

        # Container to store intermediate results
        self.gz = None
        self.dgz = None
        self.dx = None

    def forward(self, x, z, training=True):
        gz = self.generator.forward(z, training=training)
        self.gz = gz
        dgz = self.discriminator.forward(gz, training=training)
        self.dgz = dgz
        dx = self.discriminator.forward(x, training=training)
        self.dx = dx

    def backpropagation(self, momentum):
        # Backprop discriminator
        discriminator_loss = self.discriminator.loss(self.dx, self.dgz)
        error = self.discriminator.grad_loss(self.dx, self.dgz)
        for layer in self.discriminator.layers[::-1]:
            error = layer.backpropagation(error, self.learning_rate, momentum)

        # Backprop generator
        generator_loss = self.generator.loss(self.dgz)
        error = self.generator.grad_loss(error, self.dgz)
        for layer in self.generator.layers[::-1]:
            error = layer.backpropagation(error, self.learning_rate, momentum)

        return discriminator_loss, generator_loss

    def fit(self, xtrain, n_epochs=None, n_steps=None, shuffle=True, momentum=None, verbose=True):
        if verbose:
            print '\nStarting training'

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
                z = np.random.rand(self.batch_size, self.latent_dim)
                # Perform forward and backward passes
                self.forward(x, z, training=True)
                discriminator_loss, generator_loss = self.backpropagation(momentum=momentum)

                # Update printing stats and update prograss bar
                if verbose and i % max((n_steps // 20), 1) == 0:
                    rolling_discriminator_loss.append(discriminator_loss)
                    rolling_generator_loss.append(generator_loss)
                    print_progress_bar(i + 1, n_steps,
                                       prefix='Epoch %02d/%02d step %04d/%04d' % (epoch, n_epochs, i + 1, n_steps),
                                       suffix='generator loss=%.4f discriminator loss=%.4f' % (
                                           np.mean(rolling_discriminator_loss[-n_steps // 50:]),
                                           np.mean(rolling_generator_loss[-n_steps // 50:])))
                if verbose and i == n_steps - 1:
                    print_progress_bar(i + 1, n_steps,
                                       prefix='Epoch %02d/%02d step %04d/%04d' % (epoch, n_epochs, i + 1, n_steps),
                                       suffix='generator loss=%.4f discriminator loss=%.4f' % (
                                           np.mean(rolling_discriminator_loss),
                                           np.mean(rolling_generator_loss)))
                if i % 500 == 0:
                    plot_digit(None, self.gz[0], epoch*n_steps+i)
            print ' '

    def summary(self):
        print 'GENERATOR'
        self.generator.summary()
        print
        print 'DISCRIMINATOR'
        self.discriminator.summary()
