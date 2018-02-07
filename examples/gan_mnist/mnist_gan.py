__author__ = 'marvin'
from examples.mlp_mnist.mnist_loader import load_mnist
from src.models import Sequential, GAN

n_inputs = 784
n_latents = 25


def build_generator():
    mlp = Sequential(input_shape=n_latents, cost_function='gan_generator')
    mlp.add_dense_layer(100, activation='relu')
    #mlp.add_dropout_layer(dropout=.3)
    mlp.add_dense_layer(n_inputs, activation='sigmoid')
    return mlp


def build_discriminator():
    mlp = Sequential(input_shape=n_inputs, cost_function='gan_discriminator')
    mlp.add_dense_layer(100, activation='relu')
    mlp.add_dropout_layer(dropout=.3)
    mlp.add_dense_layer(1, activation='sigmoid')
    return mlp


def main():
    print('Loading train and test datasets')
    xtrain, ytrain = load_mnist('mnist', 'train')
    xtest, ytest = load_mnist('mnist', 't10k')
    print('  Done')

    gan = GAN(n_inputs, n_latents, build_generator(), build_discriminator(), batch_size=20, learning_rate=1e-4)
    gan.summary()
    gan.fit(xtrain, n_epochs=50, shuffle=True)

if __name__ == '__main__':
    main()
