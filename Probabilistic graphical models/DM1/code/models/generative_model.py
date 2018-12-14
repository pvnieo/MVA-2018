import numpy as np
from scipy.special import expit


class GenerativeModel:
    """Classifier based on the generative model.

    This class implements the generative model learned on data X and Y where:
        - X € IR², X is supposed X|Y=i ~ N(mu_i, sigma)
        - Y € {0, 1} is supposed Y ~ Bernoulli(pi)
    """

    def __init__(self, pi=0.5, mu1=(0, 0), mu0=(0, 0), sigma=np.eye(2, 2)):
        self.name = "generative model"
        self.pi = pi
        self.mu1 = np.array(mu1)
        self.mu0 = np.array(mu0)
        self.sigma = sigma
        sigma_inverse = np.linalg.inv(self.sigma)
        self.a = sigma_inverse.dot(self.mu1 - self.mu0)
        self.b = 0.5 * (self.mu0.T.dot(sigma_inverse.dot(self.mu0)) - self.mu1.T.dot(
            sigma_inverse.dot(self.mu1))) + np.log(self.pi / (1 - self.pi))

    def fit(self, X, Y):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and n_features is the number of features.
        y : array-like, shape (n_samples,)
            Target vector relative to X.
        """
        n1 = sum(Y)
        n0 = len(Y) - n1
        self.pi = n1 / (n0 + n1)
        self.mu1 = np.sum(X * np.transpose(np.array([Y, ]*2)), axis=0) / n1
        self.mu0 = np.sum(X * np.transpose(np.array([1 - Y, ]*2)), axis=0) / n0
        X1 = X - self.mu1
        X0 = X - self.mu0
        sigma_tield_1 = np.sum(X1.reshape(-1, 2, 1) * X1.reshape(-1, 1, 2) *
                               np.repeat(Y, 4, axis=0).reshape(-1, 2, 2), axis=0)
        sigma_tield_0 = np.sum(X0.reshape(-1, 2, 1) * X0.reshape(-1, 1, 2) *
                               np.repeat(1-Y, 4, axis=0).reshape(-1, 2, 2), axis=0)
        self.sigma = (sigma_tield_1 + sigma_tield_0) / (n0 + n1)
        sigma_inverse = np.linalg.inv(self.sigma)
        self.a = sigma_inverse.dot(self.mu1 - self.mu0)
        self.b = 0.5 * (self.mu0.T.dot(sigma_inverse.dot(self.mu0)) - self.mu1.T.dot(
            sigma_inverse.dot(self.mu1))) + np.log(self.pi / (1 - self.pi))

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
        a_x = np.sum(np.repeat(self.a.reshape(1, 2),
                               X.shape[0], axis=0) * X, axis=1)
        t = a_x + self.b
        C = expit(t)
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

        def line(x): return (- self.b - self.a[0] * x) / self.a[1]
        return (x, line(x))
