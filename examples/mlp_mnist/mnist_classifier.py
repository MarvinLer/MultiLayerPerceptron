__author__ = 'marvin'
from examples.mlp_mnist.mnist_loader import load_mnist
from src.multilayerperceptron import MLP

n_classes = 10

print('Loading train and test datasets')
xtrain, ytrain = load_mnist('mnist', 'train')
xtest, ytest = load_mnist('mnist', 't10k')
print('  Done')

mlp = MLP(n_inputs=784, cost_function='cross_entropy', learning_rate=5e-1, batch_size=32, random_state=123)

mlp.add_dense_layer(100, activation='relu')
mlp.add_dropout_layer(dropout=.3)
mlp.add_dense_layer(n_classes, activation='softmax')
mlp.summary()
mlp.fit(xtrain, ytrain, xtest, ytest, n_epochs=5, shuffle=True)

mlp.get_metrics(xtest, ytest)
