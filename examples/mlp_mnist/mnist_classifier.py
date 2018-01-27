__author__ = 'marvin'
from examples.mlp_mnist.mnist_loader import load_mnist
from multilayerperceptron import MLP

n_classes = 10

print('Loading train and test datasets')
xtrain, ytrain = load_mnist('mnist', 'train')
xtest, ytest = load_mnist('mnist', 't10k')
print('  Done')

#xtrain = np.asarray([[0.], [1.]])
#ytrain = np.asarray([[1., 0.], [0., 1.]])

mlp = MLP(n_inputs=784, learning_rate=5e-1,
          cost_function='cross_entropy', random_state=123)

mlp.add_layer(100, activation='sigmoid')
mlp.add_layer(n_classes, activation='softmax')
mlp.summary()
mlp.fit(xtrain, ytrain, n_steps=20000)

mlp.get_metrics(xtest, ytest)