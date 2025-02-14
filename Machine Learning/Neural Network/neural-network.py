import numpy as np
from utils import create_dataset, plot_contour
from typing import List, Dict


'''
m : no. of training examples
n : no. of features
lambdaa : regularization parameter
h1 : no. of nodes in hidden layer 1
h2 : no. of nodes in hidden layer 2
lr : learning rate
'''


class NeuralNetwork():
    def __init__(self, x, y):
        self.m, self.n = x.shape
        self.lr = 0.01
        self.lambdaa = 1e-3  # regularization factor

        self.h1 = 25  # no. of nodes in hidden layer 1
        self.h2 = 10  # o/p layer nodes = no. of classes

    def initialise_weights(self, l0, l1):
        # using 'kaiming he' weight initialisation technique
        w = np.random.randn(l0, l1) * np.sqrt(2.0/l0)
        b = np.zeros(shape=(1, l1))

        return (w, b)

    def forward_prop(self, x, parameters: Dict[str, List[float]]):
        w1 = parameters['w1']
        w2 = parameters['w2']
        b1 = parameters['b1']
        b2 = parameters['b2']

        a0 = x
        z1 = np.dot(a0, w1) + b1

        # applying activation function - ReLU
        a1 = np.maximum(0, z1)

        z2 = np.dot(a1, w2) + b2

        # applying softmax function
        exp_scores = np.exp(z2)
        probs = exp_scores/np.sum(exp_scores, axis=1, keepdims=True)
        # axis = 1 means row-wise

        cache = {'a0': a0, 'a1': a1, 'probs': probs}

        return cache, probs

    def compute_cost(self, y, probs, parameters):
        # loss = original_loss + regularization_loss
        w1 = parameters['w1']
        w2 = parameters['w2']

        y = y.astype(int)

        # cross entropy loss
        loss = np.sum(-np.log(probs[np.arange(stop=self.m), y]) / self.m)

        # l2 (ridge regularization)
        l2_loss = 0.5 * self.lambdaa * np.sum(np.power(w1, 2)) + 0.5 * self.lambdaa * np.sum(np.power(w2, 2))

        loss += l2_loss
        return loss

    def backward_prop(self, cache, parameters, y):
        # using cache from forward_prop
        a0 = cache['a0']
        a1 = cache['a1']
        probs = cache['probs']

        # unpacking the parameters
        w1 = parameters['w1']
        w2 = parameters['w2']
        b1 = parameters['b1']
        b2 = parameters['b2']

        # dJ/dz2 (here d -> curved d (dell))
        dz2 = probs
        dz2[np.arange(self.m), y] -= 1
        dz2 /= self.m

        # dJ/dw2 (here d -> curved d (dell))
        dw2 = np.dot(np.transpose(a1), dz2) + self.lambdaa * w2

        # dJ/db2 (here d -> curved d (dell))
        db2 = np.sum(dz2, axis=0, keepdims=True)

        # dJ/dz1 (here d -> curved d (dell))
        dz1 = np.dot(dz2, np.transpose(w2)) * (a1 > 0)

        # dJ/dw1 (here d -> curved d (dell))
        dw1 = np.dot(np.transpose(x), dz1) + self.lambdaa*w1

        # dJ/db1 (here d -> curved d (dell))
        db1 = np.sum(dz1, axis=0, keepdims=True)

        grads = {'dw1': dw1, 'db1': db1, 'dw2': dw2, 'db2': db2}
        return grads

    def update_parameters(self, parameters, grads):
        # gradient descent approach
        lr = self.lr

        w1 = parameters['w1']
        w2 = parameters['w2']
        b1 = parameters['b1']
        b2 = parameters['b2']

        dW2 = grads["dw2"]
        dW1 = grads["dw1"]
        db2 = grads["db2"]
        db1 = grads["db1"]

        w1 -= lr * dW1
        w2 -= lr * dW2
        b1 -= lr * db1
        b2 -= lr * db2

        parameters = {'w1': w1, 'w2': w2, 'b1': b1, 'b2': b2}
        return parameters

    def train(self, x, y, iterations=10000):
        # weight initialisation
        w1, b1 = self.initialise_weights(self.n, self.h1)
        w2, b2 = self.initialise_weights(self.h1, self.h2)

        parameters = {'w1': w1, 'w2': w2, 'b1': b1, 'b2': b2}

        for iter in range(iterations + 1):
            # forward prop
            cache, probs = self.forward_prop(x, parameters)

            # loss calculation
            loss = self.compute_cost(y, probs, parameters)

            # backward prop
            gradients = self.backward_prop(cache, parameters, y)

            # parameter updation
            parameters = self.update_parameters(parameters, gradients)

            if iter % 1000 == 0:
                print(f'Iteration: {iter}   Loss: {loss}')

        return parameters


if __name__ == '__main__':
    x, y = create_dataset()
    y = y.astype(int)

    nn = NeuralNetwork(x, y)
    trained_parameters = nn.train(x, y)

    plot_contour(x, y, nn, trained_parameters)
