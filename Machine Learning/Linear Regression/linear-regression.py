import numpy as np


'''
m : number of training examples
n : number of features
y : (1 x m)
x : (n x m)
w : (n x 1)
'''


class LinearRegression():
    def __init__(self):
        self.learning_rate = 0.001
        self.iterations = 1000

    def yhat(self, x, w):
        return np.dot(np.transpose(w), x)

    def loss(self, yhat, y):
        loss = (1/self.m) * np.sum(np.power(yhat - y, 2))  # avg mse loss
        return loss

    def gradient_descent(self, w, x, y, yhat):
        dldw = (2/self.m) * np.dot(x, np.transpose((yhat - y)))
        w -= self.learning_rate * dldw
        return w

    def main(self, x, y):
        ones = np.ones(shape=(1, x.shape[1]))
        x = np.concatenate((ones, x), axis=0)  # axis=0 means 'along rows' and axis=1 means 'along cols'
        self.m = x.shape[1]
        self.n = x.shape[0]

        w = np.zeros(shape=(self.n, 1))

        for cnt in range(self.iterations + 1):
            yhat = self.yhat(x, w)
            cost = self.loss(yhat, y)
            if cnt % 100 == 0:
                print(f'Iteration: {cnt}  Loss: {cost}')
            w = self.gradient_descent(w, x, y, yhat)

        return w


if __name__ == '__main__':
    # synthetic data -> univartiate (single feature)
    x = np.random.rand(1, 500)
    y = 3 * x + 5 + np.random.randn(1, 500) * 0.1
    regression = LinearRegression()
    w = regression.main(x, y)
