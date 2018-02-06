__author__ = 'marvin'
from examples.mlp_mnist.mnist_loader import load_mnist
from src.models import Sequential
from src.gan import GAN

n_inputs = 784
n_latents = 50


def build_generator():
    mlp = Sequential(n_inputs=n_latents, cost_function='gan_generator')
    mlp.add_dense_layer(100, activation='relu')
    #mlp.add_dropout_layer(dropout=.3)
    mlp.add_dense_layer(n_inputs, activation='sigmoid')
    return mlp


def build_discriminator():
    mlp = Sequential(n_inputs=n_inputs, cost_function='gan_discriminator')
    mlp.add_dense_layer(100, activation='relu')
    #mlp.add_dropout_layer(dropout=.3)
    mlp.add_dense_layer(1, activation='sigmoid')
    return mlp


def main():
    print('Loading train and test datasets')
    xtrain, ytrain = load_mnist('mnist', 'train')
    xtest, ytest = load_mnist('mnist', 't10k')
    print('  Done')

    gan = GAN(n_inputs, n_latents, build_generator(), build_discriminator(), batch_size=10, learning_rate=1e-1)
    gan.summary()
    gan.fit(xtrain, n_epochs=5, shuffle=True)

if __name__ == '__main__':
    main()
