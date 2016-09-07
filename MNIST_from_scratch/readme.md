# MNIST_FROM_SCRATCH
This is a single layer neural network implemented without any neural network specific library (such as tensorflow). The neural network uses chain rule to back propagate gradients and uses softmax & cross entropy to generate the loss function. I am able to achieve at least 97% accuracy on MNIST with my neural network. This experiment shows that even a single layer network works well on small dataset with few classes such as MNIST.

## Hyper parameters:
1. Learning rate: 0.5
2. Mini batch size: 10
3. Loss function: softmax + cross entropy
4. Hidden layer neurons: 200
