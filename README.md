# A handwritten Multi-Layer Perception classifier

This repository contains a custom code of a MLP classifier that depends only on numpy. Thus, this version is intended to be minimalist and easy to use in applications.

## Getting Started
### Prerequisites

```
python 2.7.10
numpy >= 1.13
```

## Running an example

The file within the folder examples/mlp_mnist contains an illustration of code using the mlp. It trains a 3-layer mlp with 784/100/10 neurons for 20000 steps using stochastic gradient descent and then retrieve loss and accuracy on the test set. To launch it:

```
python -m examples.mlp_mnist.mnist_classifier
```

from the root folder.

## Authors

* **Marvin Lerousseau** - *Ongoing work*

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* This repository is intended to be filled with various handwritten models and architectures including VAE, GAN, RNN etc

