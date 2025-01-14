from utils import *
from data.loader import loader
from model.rbm import RBM


class RBMTrainer:
    def __init__(self, v_size=15, h_size=20, lr=0.01):
        self.a, self.b, self.J = initialize_weights(v_size, h_size)
        self.v_size = v_size
        self.h_size = h_size
        self.lr = lr

    def train_rbm(self, file_path, epochs=15000):
        features, labels = loader(file_path)

        for i in range(epochs):
            # Pick a random index
            random_idx = np.random.randint(len(features))
            rbm = RBM(self.a, self.b, self.J)
            v, h = rbm.contrastive_divergence(features[random_idx])
            prob = activation_prob(v, self.J.T, self.b, T=1)

            v_true = np.concatenate((labels[random_idx], features[random_idx]), axis=0)
            self.a += self.lr * (v_true - v)
            self.b += self.lr * (prob - h)
            self.J += self.lr * (np.outer(v_true, prob) - np.outer(v, h))

            if i % 100 == 0:
                print(f'Current epoch: {i}')

        return self.a, self.b, self.J
