from model.rbm import RBM
from data.loader import loader
from model.train import RBMTrainer
from utils import *


def test_rbm_init():
    # Example usage
    a, b, J = initialize_weights()
    rbm = RBM(a, b, J)

    file_path = "../data/iris/binary_preprocessed_iris.npz"

    # Load the dataset
    features, labels = loader(file_path)

    accuracy = calculate_accuracy(rbm, features, labels)
    print(f"Accuracy: {accuracy:.2f}")


def test_rbm_trainer():
    # Initialize the RBMTrainer
    trainer = RBMTrainer()
    file_path = "../data/iris/binary_preprocessed_iris.npz"

    # Train the RBM
    print("Training the RBM...")
    a, b, J = trainer.train_rbm(file_path)

    # Create an RBM instance with the trained parameters
    rbm = RBM(a, b, J)

    # Load the dataset
    features, labels = loader(file_path)

    accuracy = calculate_accuracy(rbm, features, labels)
    print(f"Accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    test_rbm_trainer()
