import random
from utils import *
from data.loader import loader
from model.rbm import RBM


class RBMTrainer:
    """
    A class to train a Restricted Boltzmann Machine (RBM).

    Attributes:
        v_size (int): Total size of the visible layer.
        v0_size (int): Size of the initial visible input layer.
        h_size (int): Size of the hidden layer.
        lr (float): Learning rate for training.
        a (np.ndarray): Biases for the visible layer.
        b (np.ndarray): Biases for the hidden layer.
        J (np.ndarray): Weight matrix representing connections between visible and hidden layers.
    """

    def __init__(self, v_size=15, v0_size=12, h_size=6, lr=0.01):
        """
        Initializes the RBMTrainer instance with the given parameters.

        Args:
            v_size (int, optional): Total size of the visible layer. Default is 15.
            v0_size (int, optional): Size of the initial visible input layer. Default is 12.
            h_size (int, optional): Size of the hidden layer. Default is 6.
            lr (float, optional): Learning rate for training. Default is 0.01.
        """
        self.a, self.b, self.J = initialize_weights(v_size, h_size)
        self.v_size = v_size
        self.v0_size = v0_size
        self.h_size = h_size
        self.lr = lr

    def train_rbm(self, dataset_path, epochs=10000):
        """
        Trains the RBM using the provided dataset.

        Args:
            dataset_path (str): Path to the dataset file. The dataset should consist of features and labels.
            epochs (int, optional): Number of training iterations. Default is 10,000.

        Returns:
            tuple: Updated parameters (a, b, J) for the RBM.
        """
        # Load the dataset
        features, labels = loader(dataset_path)

        # Initialize the RBM instance with current parameters
        rbm = RBM(self.a, self.b, self.J, self.v_size, self.v0_size, self.h_size)

        # Training loop
        for i in range(epochs):
            # Randomly select a feature-label pair
            feature, label = random.choice(list(zip(features, labels)))

            # Combine label and feature to form the true visible vector
            v_true = np.concatenate((label, feature), axis=0)

            # Compute the hidden layer activation probabilities
            h_prob = activation_prob(v_true, self.J.T, self.b, T=1)

            # Perform a contrastive divergence step
            v, h = rbm.contrastive_divergence_step(v_true, T=1)

            # Update parameters using gradient descent
            self.a += self.lr * (v_true - v)
            self.b += self.lr * (h_prob - h)
            self.J += self.lr * (np.outer(v_true, h_prob) - np.outer(v, h))

            # Update the RBM instance with the new parameters
            rbm.update_parameters(self.a, self.b, self.J)

            # Print progress at regular intervals
            if (i + 1) % 100 == 0 or i == epochs - 1:
                print(f'Epoch: {i + 1}')

        return self.a, self.b, self.J


if __name__ == "__main__":
    # Initialize the RBMTrainer
    trainer = RBMTrainer()

    dataset_path = '../data/iris/binary_preprocessed_iris.npz'
    weights_path = '../data/iris/rbm_weights.npz'

    # Train the RBM
    print("Training the RBM...")
    a, b, J = trainer.train_rbm(dataset_path)

    # Save the weights to the specified file
    np.savez(weights_path, a=a, b=b, J=J)
    print(f"Weights saved to '{weights_path}'")
