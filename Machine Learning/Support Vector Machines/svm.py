import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

'''
Loss Function: https://medium.com/towards-data-science/optimization-loss-function-under-the-hood-part-iii-5dff33fa015d
'''


class SVM():
    def __init__(self, learning_rate=0.001, iterations=10000, regularization_param=0.01):
        self.lr = learning_rate
        self.iterations = iterations
        self.lambdaa = regularization_param
        self.w = None
        self.b = None

    def fit(self, x, y):
        m, n = x.shape
        self.w = np.zeros(shape=(n,))
        self.b = 0

        for it in range(self.iterations):
            for i, x_i in enumerate(x):
                constraint = (y[i] * (np.dot(x_i, self.w) - self.b) >= 1)
                if constraint:
                    self.w -= self.lr * (2 * self.lambdaa * self.w)  # directly implemented gradient descent approach
                else:
                    self.w -= self.lr * (2 * self.lambdaa * self.w - np.dot(x_i, y[i]))
                    self.b -= self.lr * y[i]

    def predict(self, x):
        prediction = np.dot(x, self.w) - self.b
        return np.sign(prediction)


def plot(x, y):
    plt.figure(figsize=(8, 6))
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap='bwr')
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.title('Iris Dataset (Setosa vs. Non-Setosa)')
    plt.show()


if __name__ == '__main__':
    data = load_iris()
    x = data.data[:, :2]
    y = data.target
    y = np.array(list(map(lambda x: -1 if x == 0 else 1, y)))
    # plot(x, y)

    svm = SVM()
    svm.fit(x, y)

    def plot_decision_boundary(x, y, model):
        plt.scatter(x[:, 0], x[:, 1], c=y, cmap='bwr')
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))
        xy = np.vstack([xx.ravel(), yy.ravel()]).T
        Z = model.predict(xy).reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
        plt.show()

    plot_decision_boundary(x, y, svm)
