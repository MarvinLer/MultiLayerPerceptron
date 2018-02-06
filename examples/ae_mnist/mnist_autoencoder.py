__author__ = 'marvin'
from examples.mlp_mnist.mnist_loader import load_mnist
from src.models import Sequential


def main():
    print('Loading train and test datasets')
    xtrain, ytrain = load_mnist('mnist', 'train')
    xtest, ytest = load_mnist('mnist', 't10k')
    print('  Done')

    mlp = Sequential(n_inputs=784, cost_function='squared_loss', learning_rate=1e-3, batch_size=64, random_state=123)

    mlp.add_dense_layer(50, activation='relu')
    mlp.add_dense_layer(784, activation='relu')
    mlp.summary()
    mlp.fit(xtrain, xtrain, xtest, xtest, n_epochs=5, shuffle=True)

    mlp.get_metrics(xtest, ytest)

if __name__ == '__main__':
    main()
