import numpy as np
from scipy.special import expit


class QDAModel:
    """Classifier based on the QDA model.

    This class implements the QDA model learned on data X and Y where:
        - X € IR², X is supposed X|Y=i ~ N(mu_i, sigma_i)
        - Y € {0, 1} is supposed Y ~ Bernoulli(pi)
    """

    def __init__(self, pi=0.5, mu0=(0, 0), mu1=(0, 0),
                 sigma0=np.eye(2, 2), sigma1=np.eye(2, 2)):
        self.name = "qda model"
        self.pi = pi
        self.mu1 = np.array(mu1)
        self.mu0 = np.array(mu0)
        self.sigma0 = sigma0
        self.sigma1 = sigma1
        sigma_inverse1 = np.linalg.inv(self.sigma1)
        sigma_inverse0 = np.linalg.inv(self.sigma0)
        self.q = sigma_inverse0 - sigma_inverse1
        self.a = sigma_inverse1.dot(self.mu1) - sigma_inverse0.dot(self.mu0)
        self.b = 0.5 * (self.mu0.T.dot(sigma_inverse0.dot(self.mu0)) - self.mu1.T.dot(sigma_inverse1.dot(self.mu1))) + np.log(
            self.pi / (1 - self.pi)) + 0.5 * np.log(np.linalg.det(sigma_inverse0) / np.linalg.det(sigma_inverse1))

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
        self.mu1 = np.sum(X * np.array([Y, ] * 2).T, axis=0) / n1
        self.mu0 = np.sum(X * np.array([1 - Y, ] * 2).T, axis=0) / n0
        X1 = X - self.mu1
        X0 = X - self.mu0
        sigma_tield_1 = np.sum(X1.reshape(-1, 2, 1) * X1.reshape(-1, 1, 2) *
                               np.repeat(Y, 4, axis=0).reshape(-1, 2, 2), axis=0)
        sigma_tield_0 = np.sum(X0.reshape(-1, 2, 1) * X0.reshape(-1, 1, 2) *
                               np.repeat(1 - Y, 4, axis=0).reshape(-1, 2, 2), axis=0)
        self.sigma0 = sigma_tield_0 / n0
        self.sigma1 = sigma_tield_1 / n1
        sigma_inverse1 = np.linalg.inv(self.sigma1)
        sigma_inverse0 = np.linalg.inv(self.sigma0)
        self.q = sigma_inverse0 - sigma_inverse1
        self.a = sigma_inverse1.dot(self.mu1) - sigma_inverse0.dot(self.mu0)
        self.b = 0.5 * (self.mu0.T.dot(sigma_inverse0.dot(self.mu0)) - self.mu1.T.dot(sigma_inverse1.dot(self.mu1))) + np.log(
            self.pi / (1 - self.pi)) + 0.5 * np.log(np.linalg.det(self.sigma0) / np.linalg.det(self.sigma1))

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
        q_x_2 = 0.5 * np.apply_along_axis(lambda x: x.dot(self.q.dot(x)), 1, X)
        a_x = np.sum(np.repeat(self.a.reshape(1, 2),
                               X.shape[0], axis=0) * X, axis=1)
        t = q_x_2 + a_x + self.b
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
        xmax, xmin, ymax, ymin = -10, 10, -10, 10
        step = 400
        X, Y = np.meshgrid(np.arange(xmin, xmax, (xmax - xmin) * 1. / step), np.arange(ymin, ymax, (ymax - ymin) * 1. / step))
        grid = np.c_[X.ravel(), Y.ravel()]
        return X, Y, grid
