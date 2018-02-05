import numpy as np
from src.multilayerperceptron import MLP


class GAN(object):
    def __init__(self, n_inputs):
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.n_inputs = 784
        self.latent_dim = 50

    def build_generator():
        mlp = MLP(n_inputs=self.latent_dim)
        mlp.add_dense_layer(256, activation='relu')
        mlp.add_dropout_layer(dropout=.3)
        mlp.add_dense_layer(self.n_inputs, activation='relu')
        return mlp

    def build_discriminator():
        mlp = MLP(n_inputs=self.n_inputs)
        mlp.add_dense_layer(256, activation='relu')
        mlp.add_dropout_layer(dropout=.3)
        mlp.add_dense_layer(1, activation='sigmoid')
        return mlp

    def loss_1():

        return

    def loss_2():



    def fit(self, X, y, n_epochs):
        for epoch in range(n_epochs):
            # Train Discriminator
            random = np.random.randn(self.n_inputs)
            


            # Train Generator




