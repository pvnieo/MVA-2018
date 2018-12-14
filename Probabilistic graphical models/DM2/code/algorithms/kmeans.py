import numpy as np
from random import uniform
from scipy.spatial.distance import euclidean


class Kmeans():
    """Implements K-Means clustering algorithm

    Parameters
    ----------
    n_clusters : int, optional, default: 4
        The number of clusters to form as well as the number of
        centroids to generate.
    max_iter : int, default: 300
        Maximum number of iterations of the k-means algorithm for a single run.

    Attributes
    ----------
    cluster_centers : array, [n_clusters, n_features]
        Coordinates of cluster centers
    labels :
        Labels of each point
    distortion :
        Distortion for the learned data
    """

    def __init__(self, n_clusters=4, max_iter=300):
        self.name = "kmeans"
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.cluster_centers = None
        self.labels = None
        self.distortion = -1

    def _closest_cluster(self, x):
        """Returns the ID of the closest cluster to x

        Parameters
        ----------
        x : array-like, shape=(n_features)
            Data point.

        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the closest cluster for each sample.
        """
        distances = [euclidean(x, self.cluster_centers[i]) for i in range(self.n_clusters)]
        return distances.index(min(distances))

    def _distortion(self, X):
        """Compute the distortion for the learned data.

        Parameters
        ----------
        x : array-like, shape=(n_features)
            Data point.
        """
        J = 0
        for i, x in enumerate(X):
            J += euclidean(x, self.cluster_centers[self.labels[i]]) ** 2
        self.distortion = J

    def fit(self, X):
        """Compute k-means clustering.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            Training instances to cluster.
        """
        _, n_features = X.shape
        mins_feature = [np.min(X[:, i]) / 2 for i in range(n_features)]
        maxs_feature = [np.max(X[:, i]) / 2 for i in range(n_features)]
        self.cluster_centers = np.array([[uniform(mins_feature[f], maxs_feature[f]) for f in range(n_features)]
                                        for i in range(self.n_clusters)])
        iter = 0
        epsilon = 1e-6
        while iter < self.max_iter:
            self.labels = np.array([self._closest_cluster(x) for x in X])
            centers = np.array([np.mean(X[(self.labels == i)], axis=0) for i in range(self.n_clusters)])
            if (np.abs(centers - self.cluster_centers) < epsilon).all():
                self.cluster_centers = centers
                break
            self.cluster_centers = centers
            iter += 1
        self._distortion(X)

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            New data to predict.

        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        return np.array([self._closest_cluster(x) for x in X])
