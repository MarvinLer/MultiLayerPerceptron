__author__ = 'marvin'
from mnist_loader import load_mnist
from multilayerperceptron import MLP

n_classes = 10

print('Loading train and test datasets')
#xtrain, ytrain = load_mnist('mnist', 'train')
xtest, ytest = load_mnist('mnist', 't10k')
print('  Done')

mlp = MLP(n_inputs=28*28, learning_rate=1e-1,
          cost_function='cross_entropy', random_state=123)

mlp.add_layer(100)
mlp.add_layer(n_classes, activation=None)
mlp.summary()
mlp.fit(xtest, ytest, n_steps=10000)