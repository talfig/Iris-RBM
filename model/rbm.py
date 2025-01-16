from utils import *


class RBM:
    """
    A class to implement a Restricted Boltzmann Machine (RBM).
    """

    def __init__(self, a, b, J, v_size=15, v0_size=12, h_size=6, T=10):
        self.a = a
        self.b = b
        self.J = J
        self.T = T

        self.v_size = v_size
        self.v0_size = v0_size
        self.h_size = h_size

    def energy(self, v, h):
        """
        Compute the energy function E(v, h).

        Parameters:
            v (numpy.ndarray): Vector of visible units (size 15).
            h (numpy.ndarray): Vector of hidden units (size 6).

        Returns:
            float: Energy value.
        """
        term1 = -np.dot(self.a, v)  # -sum(a_i * v_i)
        term2 = -np.dot(self.b, h)  # -sum(b_j * h_j)
        term3 = -np.dot(v, np.dot(self.J, h))  # -sum(J_ij * v_i * h_j)

        return term1 + term2 + term3

    def update_parameters(self, a, b, J):
        self.a = a
        self.b = b
        self.J = J

    def sample_units(self, units, weights, bias):
        prob = activation_prob(units, weights, bias, self.T)

        # Iterate over every group of three elements
        for i in range(0, len(prob), 3):
            # Get the current group of three
            group = prob[i:i + 3]

            # Find the index of the highest probability in the current group
            max_index = np.argmax(group)

            # Set all probabilities in the group to zero except the highest one
            group[:] = 0
            group[max_index] = 1  # Set the highest probability spot to 1

            # Update the original probability vector
            prob[i:i + 3] = group

        return (np.random.rand(len(prob)) < prob).astype(int)

    def sample_hidden(self, v):
        return self.sample_units(v, self.J.T, self.b)

    def sample_visible(self, h):
        return self.sample_units(h, self.J, self.a)

    def generate_visible_layer(self):
        # Generate the visible layer of unfrozen neurons (v) with 0s and 1s
        v = np.random.randint(0, 2, self.v_size - self.v0_size)
        return v

    def generate_hidden_layer(self):
        # Generate the hidden layer of neurons (h) with 0s and 1s
        h = np.random.randint(0, 2, self.h_size)
        return h

    def contrastive_divergence(self, v0, k=1000):
        """Performs contrastive divergence with multiple steps."""
        unfrozen_v, h = self.generate_visible_layer(), self.generate_hidden_layer()
        v = np.concatenate((unfrozen_v, v0), axis=0)

        for _ in range(k):
            v, h = self.contrastive_divergence_step(v)
            v = np.concatenate((v[:self.v_size - self.v0_size], v0), axis=0)
            self.T -= 0.1

        return v[:self.v_size - self.v0_size]

    def contrastive_divergence_step(self, v):
        """Performs a single step of contrastive divergence."""
        h = self.sample_hidden(v)
        v = self.sample_visible(h)
        return v, h
