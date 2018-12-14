import numpy as np
from scipy.special import expit
from scipy.spatial.distance import euclidean
from math import sqrt, pi


class LinearRegression:
    """Classifier based on the linear regression model.

    This class implements the linear regression model learned on data X and Y where:
        - X € IR²
        - Y € {0, 1}
    """

    def __init__(self, W=(0, 0, 0)):
        self.name = "linear regression"
        self.W = np.array(W)

    def fit(self, X, Y):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and n_features is the number of features.
        y : array-like, shape (n_samples,)
            Target vector relative to X.
        """
        X_p = np.ones((X.shape[0], X.shape[1]+1))
        X_p[:, :-1] = X
        self.W = np.linalg.pinv(X_p).dot(Y)

    def predict(self, X, threshold=0.5):
        """Predict using the trained model

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Samples.
        Returns
        -------
        C : array, shape (n_samples,)
            Returns predicted classes.
        """
        X_p = np.ones((X.shape[0], X.shape[1]+1))
        X_p[:, :-1] = X
        C = np.dot(X_p, self.W)
        C = np.array([1 if x else 0 for x in C >= threshold])
        return C

    def score(self, X, Y):
        """
        Return the correct percentage of classification
        """
        return np.sum(self.predict(X) == Y) / len(Y)

    def get_boundary(self, data):
        """return point to plot the boundary p(y = 1|x) = 0.5"""
        x = np.linspace(np.min(data.x1) - 1, np.max(data.x1) + 1, 10)

        def line(x): return (0.5 - self.W[2] - self.W[0] * x) / self.W[1]
        return (x, line(x))
