import numpy as np


def sigmoid(x):
    """
    Sigmoid activation function.

    Parameters:
        x (numpy.ndarray): Input array.

    Returns:
        numpy.ndarray: Sigmoid output.
    """
    return 1 / (1 + np.exp(-x))


# Function to initialize random weights
def initialize_weights(v_size=15, h_size=6):
    a = np.random.randn(v_size)  # Random vector for 'a' of size v_size
    b = np.random.randn(h_size)  # Random vector for 'b' of size h_size
    J = np.random.randn(v_size, h_size)  # Random matrix for 'J' of size (v_size, h_size)

    return a, b, J


def activation_prob(units, weights, bias, T=1):
    activation = np.dot(units, weights.T) + bias
    prob = sigmoid(activation / T)
    return prob


def calculate_accuracy(rbm, features, labels):
    """Calculate accuracy of the RBM for the given features and labels."""
    correct = 0
    for feature, label in zip(features, labels):
        prediction = rbm.contrastive_divergence(feature)
        if np.array_equal(prediction, label):
            correct += 1

    return correct / len(labels)
