import numpy as np
from sklearn.datasets import make_blobs

'''
m : number of training examples
n : number of features
x : m x n
y : m x 1
w : n x 1
'''


class LogisticRegression():
    def __init__(self, learning_rate=0.01, iterations=10000):
        self.learningrate = learning_rate
        self.iterations = iterations

    def train(self, x, y):
        m, n = x.shape
        self.weights = np.zeros(shape=(n, 1))
        self.bias = 0

        for it in range(self.iterations):
            p_predict = self.sigmoid(np.dot(x, self.weights) + self.bias)
            loss = -1/m * np.sum(y * np.log(p_predict) + (1-y)*np.log(1-p_predict))

            dw = 1/m * np.dot(np.transpose(x), (p_predict - y))
            db = 1/m * np.sum(p_predict - y)

            self.weights -= self.learningrate * dw
            self.bias -= self.learningrate * db

            if it % 1000 == 0:
                print(f'iteration: {it} loss: {loss}')

        return (self.weights, self.bias)

    def sigmoid(self, z):
        p = 1 / (1 + np.exp(-z))
        return p

    def predict(self, x):
        p_predict = self.sigmoid(np.dot(x, self.weights) + self.bias)
        return (p_predict >= 0.5).astype(int)


if __name__ == '__main__':
    np.random.seed(5)
    x, y = make_blobs(n_samples=1000, n_features=2, centers=2)  # centers correspond to 'class' like data is centered around two clusters
    y = y[:, np.newaxis]

    model = LogisticRegression()
    w, b = model.train(x, y)
    y_predict = model.predict(x)

    print(f'accuracy: {np.sum(y == y_predict)/x.shape[0]}')
