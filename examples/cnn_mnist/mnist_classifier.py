__author__ = 'marvin'
from examples.mlp_mnist.mnist_loader import load_mnist
from src.models import Sequential

n_classes = 10
batch_size = 5

print('Loading train and test datasets')
xtrain, ytrain = load_mnist('mnist', 'train', reshape2d=True)
xtest, ytest = load_mnist('mnist', 't10k', reshape2d=True)
print xtrain.shape
print('  Done')

mlp = Sequential(input_shape=(28, 28, 1), cost_function='cross_entropy', learning_rate=5e-1, batch_size=batch_size,
                 random_state=123)

mlp.add_conv2d_layer(filter_size=(3, 3), n_filters=10, stride=(1, 1), padding=None, activation='relu')
mlp.add_reshape_layer(newshape=(-1,))
mlp.add_dense_layer(n_classes, activation='softmax')
mlp.summary()
mlp.fit(xtrain, ytrain, xtest, ytest, n_epochs=5, shuffle=True)

mlp.get_metrics(xtest, ytest)
