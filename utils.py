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


def initialize_weights(v_size=15, h_size=6):
    """
    Initialize random weights for an RBM (Restricted Boltzmann Machine).

    Parameters:
        v_size (int): Size of the visible layer (default is 15).
        h_size (int): Size of the hidden layer (default is 6).

    Returns:
        tuple: A tuple containing:
            - a (numpy.ndarray): Random vector for visible layer biases of size v_size.
            - b (numpy.ndarray): Random vector for hidden layer biases of size h_size.
            - J (numpy.ndarray): Random matrix for weights connecting visible and hidden layers of size (v_size, h_size).
    """
    a = np.random.randn(v_size)  # Random vector for 'a' of size v_size
    b = np.random.randn(h_size)  # Random vector for 'b' of size h_size
    J = np.random.randn(v_size, h_size)  # Random matrix for 'J' of size (v_size, h_size)

    return a, b, J


def activation_prob(units, weights, bias, T=1):
    """
    Compute the activation probabilities for the given units, weights, and biases.

    Parameters:
        units (numpy.ndarray): Input units (visible or hidden layer values).
        weights (numpy.ndarray): Weights matrix (connections between layers).
        bias (numpy.ndarray): Bias vector.
        T (float): Temperature parameter for scaling (default is 1).

    Returns:
        numpy.ndarray: Activation probabilities after applying the sigmoid function.
    """
    activation = np.dot(units, weights.T) + bias
    prob = sigmoid(activation / T)
    return prob


def calculate_accuracy(rbm, features, labels):
    """
    Calculate the accuracy of the RBM for the given features and labels.

    Parameters:
        rbm (RBM): The RBM model instance.
        features (list or numpy.ndarray): List or array of input features.
        labels (list or numpy.ndarray): List or array of true labels corresponding to the features.

    Returns:
        float: The accuracy of the RBM model, calculated as the fraction of correctly predicted labels.
    """
    correct = 0
    for feature, label in zip(features, labels):
        prediction = rbm.contrastive_divergence(feature)
        if np.array_equal(prediction, label):
            correct += 1

    return correct / len(labels)
