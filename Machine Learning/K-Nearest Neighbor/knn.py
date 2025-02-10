import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


class KNearestNeighbor():
    def __init__(self, k):
        self.k = k

    def train(self, x, y):
        self.x_train = x
        self.y_train = y

    def predict(self, x_test):
        # calculate distance for each test_data wrt to train_data (dim(x_test) x dim(x_train)) -> like in floyd-warshall algo
        distances = self.distance(x_test)

        # label predict
        num_test = x_test.shape[0]
        y_pred = []

        for i in range(num_test):
            y_indices = np.argsort(distances[i, :])
            k_closest_class = self.y_train[y_indices[:self.k]].astype(int)
            y_pred.append(np.argmax(np.bincount(k_closest_class)))

        return y_pred

    def distance(self, x_test, num_loops: int = 0):
        num_test = x_test.shape[0]
        num_train = self.x_train.shape[0]

        distances = np.zeros(shape=(num_test, num_train))

        # calculating euclidean distances

        '''
        using nested loops (brute force method)
        for i in range(num_test):
            for j in range(num_train):
                distances[i][j] = np.sqrt((x_test[i][0] - self.x_train[j][0])**2 + (x_test[i][1] - self.x_train[j][1])**2)

        '''
        # using vectorized multiplication method (optimized)
        x_test = np.array(x_test)
        x_train = np.array(self.x_train)
        test_sq = np.sum(x_test ** 2, axis=1).reshape(num_test, 1)
        train_sq = np.sum(x_train * 2, axis=1).reshape(1, num_test)
        distances = test_sq + train_sq - 2 * np.dot(x_test, np.transpose(self.x_train))
        # should have takes sqrt of the above value for actual distances -> but no issue if we dont take

        return distances


def plot(X, y):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y == 0, 0], X[y == 0, 1], color='blue', label='Class 0', marker='o')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', label='Class 1', marker='s')
    plt.grid(True)
    plt.show()


def generate_dataset():
    x, y = make_blobs(n_samples=100, centers=2, n_features=2, cluster_std=1.5, random_state=1)
    return (x, y)


if __name__ == '__main__':
    x, y = generate_dataset()
    plot(x, y)

    knn = KNearestNeighbor(1)
    knn.train(x, y)

    y_pred = knn.predict(x)
    print(f'Accuracy : {sum(y_pred == y) / len(y)}')
