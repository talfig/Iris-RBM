from utils import *


class RBM:
    """
    A class to implement a Restricted Boltzmann Machine (RBM).
    """

    def __init__(self, a, b, J, v_size=15, v0_size=12, h_size=6, T_init=10, T_rate=0.1):
        self.a = a
        self.b = b
        self.J = J
        self.T_init = T_init
        self.T_rate = T_rate

        self.v_size = v_size
        self.v0_size = v0_size
        self.h_size = h_size

    def update_parameters(self, a, b, J):
        self.a = a
        self.b = b
        self.J = J

    def sample_hidden(self, v, T=1):
        h_prob = activation_prob(v, self.J.T, self.b, T)
        return (np.random.rand(len(h_prob)) < h_prob).astype(float)

    def sample_visible(self, h, T=1):
        v_prob = activation_prob(h, self.J, self.a, T)
        return (np.random.rand(len(v_prob)) < v_prob).astype(float)

    def generate_visible_layer(self):
        # Generate the visible layer of unfrozen neurons (v) with 0s and 1s
        v = np.random.randint(0, 2, self.v_size - self.v0_size).astype(float)
        return v

    def generate_hidden_layer(self):
        # Generate the hidden layer of neurons (h) with 0s and 1s
        h = np.random.randint(0, 2, self.h_size).astype(float)
        return h

    def contrastive_divergence(self, v0, k=100):
        """Performs contrastive divergence with multiple steps."""
        unfrozen_v, h = self.generate_visible_layer(), self.generate_hidden_layer()
        v = np.concatenate((unfrozen_v, v0), axis=0)
        T = self.T_init

        for _ in range(k):
            v, h = self.contrastive_divergence_step(v, T)
            v[self.v_size - self.v0_size:] = v0
            T -= self.T_rate

        return v[:self.v_size - self.v0_size]

    def contrastive_divergence_step(self, v, T=1):
        """Performs a single step of contrastive divergence."""
        h = self.sample_hidden(v, T)
        v = self.sample_visible(h, T)
        return v, h
