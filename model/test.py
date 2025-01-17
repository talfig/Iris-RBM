from model.rbm import RBM
from data.loader import loader
from utils import *


def test_rbm_init(dataset_path):
    # Example usage
    a, b, J = initialize_weights()
    rbm = RBM(a, b, J)

    # Load the dataset
    features, labels = loader(dataset_path)

    accuracy = calculate_accuracy(rbm, features, labels)
    print(f"Accuracy: {accuracy:.2f}")


def test_rbm_trainer(dataset_path, weights_path):
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

    accuracy = calculate_accuracy(rbm, features, labels)
    print(f"Accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    dataset_path = '../data/iris/binary_preprocessed_iris.npz'
    weights_path = '../data/iris/rbm_weights.npz'

    print("Starting evaluation of model initialization:")
    test_rbm_init(dataset_path)

    print("Starting evaluation of model training:")
    test_rbm_trainer(dataset_path, weights_path)
