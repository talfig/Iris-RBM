from model.rbm import RBM
from data.loader import loader
from utils import *


def test_rbm_init(dataset_path):
    """
    Test the initialization of an RBM (Restricted Boltzmann Machine) and evaluate its performance.

    Parameters:
        dataset_path (str): Path to the dataset file to load features and labels.

    Example Usage:
        test_rbm_init("path/to/dataset.npz")

    Workflow:
        1. Initialize the RBM with random weights and biases using `initialize_weights`.
        2. Load the dataset using the `loader` function to obtain features and labels.
        3. Calculate the accuracy of the RBM using `calculate_accuracy`.
        4. Print the calculated accuracy.
    """
    # Example usage
    a, b, J = initialize_weights()
    rbm = RBM(a, b, J)

    # Load the dataset
    features, labels = loader(dataset_path)

    # Calculate accuracy
    accuracy = calculate_accuracy(rbm, features, labels)
    print(f"Accuracy: {accuracy:.2f}")


def test_rbm_trainer(dataset_path, weights_path):
    """
    Test the performance of a trained RBM using pre-saved weights.

    Parameters:
        dataset_path (str): Path to the dataset file to load features and labels.
        weights_path (str): Path to the file containing the pre-trained RBM weights.

    Example Usage:
        test_rbm_trainer("path/to/dataset.npz", "path/to/weights.npz")

    Workflow:
        1. Load the pre-trained weights from the specified file using `np.load`.
        2. Extract the visible layer bias (`a`), hidden layer bias (`b`), and weight matrix (`J`).
        3. Create an RBM instance with the extracted weights.
        4. Load the dataset using the `loader` function to obtain features and labels.
        5. Calculate the accuracy of the RBM using `calculate_accuracy`.
        6. Print the calculated accuracy.
    """
    # Load the trained RBM weights from the specified file
    weights = np.load(weights_path)

    # Extract the weights and parameters from the saved file
    a = weights['a']  # Visible layer bias
    b = weights['b']  # Hidden layer bias
    J = weights['J']  # Weight matrix

    # Create an RBM instance with the trained parameters
    rbm = RBM(a, b, J)

    # Load the dataset
    features, labels = loader(dataset_path)

    # Calculate accuracy
    accuracy = calculate_accuracy(rbm, features, labels)
    print(f"Accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    dataset_path = '../data/iris/binary_preprocessed_iris.npz'
    weights_path = '../data/iris/rbm_weights.npz'

    print("Starting evaluation of model initialization:")
    test_rbm_init(dataset_path)

    print("Starting evaluation of model training:")
    test_rbm_trainer(dataset_path, weights_path)
