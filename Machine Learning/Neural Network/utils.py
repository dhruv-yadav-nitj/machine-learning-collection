import numpy as np
import matplotlib.pyplot as plt


def create_dataset(n=100, k=2):
    d = 2  # n data instances in each class -> total data = n*k and d -> no. of features
    x = np.zeros(shape=(n*k, d))
    y = np.zeros(shape=(n*k))

    for j in range(k):
        index = range(n * j, n * (j+1))
        r = np.linspace(0, 1, n)
        t = np.linspace(j * 4, (j + 1) * 4, n) + np.random.randn(n) * 0.2
        x[index] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[index] = j

    plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    # plt.show()

    return (x, y)


def plot_contour(X, y, model, parameters):
    # plot the resulting classifier
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    points = np.c_[xx.ravel(), yy.ravel()]

    # forward prop with our trained parameters
    _, Z = model.forward_prop(points, parameters)

    # classify into highest prob
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)

    # plt the points
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.show()
    # fig.savefig('spiral_net.png')
