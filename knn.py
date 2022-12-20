import numpy as np
from scipy.stats import mode


class KNN:
    def __init__(self, k):
        """
        constructor
        :param k: k used for knn model
        """
        self.X_train = None
        self.y_train = None
        self.k = k
        self.test_dist_idxs = None

    def fit(self, X_train, y_train, k):
        """
        calc mean and std
        :param X_train: calc stats for this param
        :param y_train: and this one too
        :return:
        """
        self.X_train = X_train
        self.y_train = y_train
        self.k = k

    def predict(self, X_test):
        pred = []
        for i, x in enumerate(X_test):
            if (self.test_dist_idxs is None) or (i not in self.test_dist_idxs.keys()):
                indices = self.neighbours_indices(x, i)  # index's of closest neighbors
            else:
                indices = self.test_dist_idxs[i][:self.k]  # distances where already computed
            closest_labels = [self.y_train[index] for index in indices]  # labels of closest neighbors
            predicted_label = (mode(closest_labels)).mode[0]
            pred.append(predicted_label)  # get the mode and append to the list
        return pred

    def neighbours_indices(self, x, i):
        """
        find indices of k closest neighbors to x
        :param x: the point
        :return: np array of said indices
        """
        distances = []
        for sample in self.X_train:  # get distances of all point from x
            dist = KNN.dist(x, sample)
            distances.append(dist)
        idx = np.argsort(np.array(distances))  # get only the first k
        if self.test_dist_idxs is None:
            self.test_dist_idxs = dict()
            self.test_dist_idxs[i] = idx
        else:
            self.test_dist_idxs[i] = idx
        return idx[:self.k]

    @staticmethod
    def dist(x1, x2):
        """
        euclidian distance method
        :param x1: np array
        :param x2: np array
        :return: euclidian distance between 2 vectors x1 and x2
        """
        return np.linalg.norm(x1 - x2)
