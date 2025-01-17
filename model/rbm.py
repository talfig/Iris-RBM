from utils import *


class RBM:
    """
    A class to implement a Restricted Boltzmann Machine (RBM).

    Attributes:
        a (np.ndarray): Biases for the visible layer.
        b (np.ndarray): Biases for the hidden layer.
        J (np.ndarray): Weight matrix representing connections between visible and hidden layers.
        T_init (float): Initial temperature parameter for contrastive divergence.
        T_rate (float): Decrease rate of the temperature parameter.
        v_size (int): Total size of the visible layer.
        v0_size (int): Size of the frozen part of the visible layer.
        h_size (int): Size of the hidden layer.
    """

    def __init__(self, a, b, J, v_size=15, v0_size=12, h_size=6, T_init=10, T_rate=0.1):
        """
        Initializes the RBM instance with given parameters.

        Args:
            a (np.ndarray): Biases for the visible layer.
            b (np.ndarray): Biases for the hidden layer.
            J (np.ndarray): Weight matrix for visible-to-hidden connections.
            v_size (int, optional): Total size of the visible layer. Default is 15.
            v0_size (int, optional): Size of the frozen part of the visible layer. Default is 12.
            h_size (int, optional): Size of the hidden layer. Default is 6.
            T_init (float, optional): Initial temperature parameter. Default is 10.
            T_rate (float, optional): Temperature decrease rate. Default is 0.1.
        """
        self.a = a
        self.b = b
        self.J = J
        self.T_init = T_init
        self.T_rate = T_rate

        self.v_size = v_size
        self.v0_size = v0_size
        self.h_size = h_size

    def update_parameters(self, a, b, J):
        """
        Updates the biases and weight matrix for the RBM.

        Args:
            a (np.ndarray): New biases for the visible layer.
            b (np.ndarray): New biases for the hidden layer.
            J (np.ndarray): New weight matrix for visible-to-hidden connections.
        """
        self.a = a
        self.b = b
        self.J = J

    def sample_hidden(self, v, T=1):
        """
        Samples the hidden layer states based on the visible layer and temperature.

        Args:
            v (np.ndarray): Current visible layer states.
            T (float, optional): Temperature parameter. Default is 1.

        Returns:
            np.ndarray: Binary states of the hidden layer.
        """
        h_prob = activation_prob(v, self.J.T, self.b, T)
        return (np.random.rand(len(h_prob)) < h_prob).astype(float)

    def sample_visible(self, h, T=1):
        """
        Samples the visible layer states based on the hidden layer and temperature.

        Args:
            h (np.ndarray): Current hidden layer states.
            T (float, optional): Temperature parameter. Default is 1.

        Returns:
            np.ndarray: Binary states of the visible layer.
        """
        v_prob = activation_prob(h, self.J, self.a, T)
        return (np.random.rand(len(v_prob)) < v_prob).astype(float)

    def generate_visible_layer(self):
        """
        Generates the unfrozen part of the visible layer.

        Returns:
            np.ndarray: Binary states for the unfrozen visible neurons.
        """
        v = np.random.randint(0, 2, self.v_size - self.v0_size).astype(float)
        return v

    def generate_hidden_layer(self):
        """
        Generates the hidden layer states.

        Returns:
            np.ndarray: Binary states for the hidden neurons.
        """
        h = np.random.randint(0, 2, self.h_size).astype(float)
        return h

    def contrastive_divergence(self, v0, k=100):
        """
        Performs contrastive divergence to train the RBM.

        Args:
            v0 (np.ndarray): Initial states of the frozen part of the visible layer.
            k (int, optional): Number of steps for contrastive divergence. Default is 100.

        Returns:
            np.ndarray: Updated states for the unfrozen visible layer.
        """
        unfrozen_v, h = self.generate_visible_layer(), self.generate_hidden_layer()
        v = np.concatenate((unfrozen_v, v0), axis=0)
        T = self.T_init

        for _ in range(k):
            v, h = self.contrastive_divergence_step(v, T)
            v[self.v_size - self.v0_size:] = v0
            T -= self.T_rate

        return v[:self.v_size - self.v0_size]

    def contrastive_divergence_step(self, v, T=1):
        """
        Performs a single step of contrastive divergence.

        Args:
            v (np.ndarray): Current visible layer states.
            T (float, optional): Temperature parameter. Default is 1.

        Returns:
            tuple: Updated visible and hidden layer states.
        """
        h = self.sample_hidden(v, T)
        v = self.sample_visible(h, T)
        return v, h
