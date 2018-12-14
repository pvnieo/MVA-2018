# project
from .kmeans import Kmeans
# 3p
import numpy as np
from random import uniform
from scipy.spatial.distance import euclidean
from scipy.stats import multivariate_normal
from numpy.linalg import norm


class EMGaussians():
    """Implements EM algorithm for Gaussian mixture model

    Parameters
    ----------
    n_gaussians : int, optional, default: 4
        The number of gaussians to form as well as the number of
        means and covariance matrix to generate.
    max_iter : int, default: 500
        Maximum number of iterations of the EM algorithm for a single run.
    kmeans_ini : bool, default: True
        Indicates whether the initialization of the means is done with kmeans
    covariance_type : {‘iso’, ‘general’}, default: ‘iso’
        Type of covariance matrix
        ‘iso’ : isotropic matrix, alpha * I
        ‘general’ : matrix without any typical structure

    Attributes
    ----------
    means : array, [n_clusters, n_features]
        Coordinates of cluster centers
    latent_var: array, [n_sample]:
        Latent variable of each point in the training data
    normalized_log_likelihood: float
        The normalized log_likelihood of the training data
    """

    def __init__(self, n_gaussians=4, max_iter=500, kmeans_ini=True, covariance_type="iso"):
        self.n_gaussians = n_gaussians
        self.max_iter = max_iter
        self.kmeans_ini = kmeans_ini
        self.covariance_type = covariance_type
        self.pi = None
        self.means = None
        self.sigmas = None
        self.tau = None
        self.latent_var = None
        self.log_likelihood = None

    def _initialisation(self, X):
        """Initialize pi, means and sigmas"""
        n_samples, n_features = X.shape
        self.log_likelihood = [-float("inf")]
        if self.kmeans_ini:
            J = float("inf")
            for _ in range(5):
                kmeans = Kmeans(n_clusters=self.n_gaussians)
                kmeans.fit(X)
                if kmeans.distortion < J:
                    J = kmeans.distortion
                    self.means = kmeans.cluster_centers.reshape(
                        self.n_gaussians, 1, -1)
        else:
            mins_feature = [np.min(X[:, i]) / 2 for i in range(n_features)]
            maxs_feature = [np.max(X[:, i]) / 2 for i in range(n_features)]
            self.means = np.array([[uniform(mins_feature[f], maxs_feature[f]) for f in range(n_features)]
                                   for i in range(self.n_gaussians)])
        self.pi = np.random.dirichlet(np.ones(4), size=1).reshape(-1, 1)
        self.sigmas = np.array([np.eye(n_features)
                                for _ in range(self.n_gaussians)])

    def _compute_tau(self, X, n_samples):
        """Compute one E-step"""
        gaussians = np.array([multivariate_normal(self.means[i].reshape(-1),
                                                  self.sigmas[i]
                                                  ).pdf(X) for i in range(self.n_gaussians)])
        self.tau = gaussians * self.pi
        self.log_likelihood.append(np.log(self.tau.sum(axis=0)).sum())
        self.tau = self.tau / self.tau.sum(axis=0)
        self.tau = self.tau.reshape(self.n_gaussians, n_samples, 1)
        return self.tau

    def _m_step(self, X, n_samples, n_features):
        """Compute one M-step"""
        self.pi = self.tau.sum(axis=1) / n_samples
        self.means = (X * self.tau).sum(axis=1) / self.tau.sum(axis=1)
        self.means = self.means.reshape(self.n_gaussians, 1, -1)
        if self.covariance_type == 'iso':
            self.sigmas = 0.5 * (((np.linalg.norm(X - self.means, axis=2) ** 2
                                   * self.tau.reshape(self.n_gaussians, -1)).sum(axis=1)
                                  / (self.tau.reshape(self.n_gaussians, -1).sum(axis=1))).reshape(self.n_gaussians, 1, 1))
            self.sigmas = self.sigmas * np.identity(self.means.shape[-1])
        elif self.covariance_type == 'general':
            self.sigmas = np.matmul(np.transpose(X - self.means, (0, 2, 1)), (X - self.means)
                                    * self.tau) / self.tau.sum(axis=1).reshape(self.n_gaussians, 1, 1)

    def fit(self, X):
        """Compute EM means and convariance matrix.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            Training instances to EM algorithm.
        """
        n_samples, n_features = X.shape
        self._initialisation(X)
        epsilon = 1e-10
        it = 1
        while it < self.max_iter:
            it += 1
            self._compute_tau(X, n_samples)
            self._m_step(X, n_samples, n_features)
            if (self.log_likelihood[-1] - self.log_likelihood[-2]) < epsilon:
                break
        self.latent_var = np.argmax(
            (self.tau.reshape(self.n_gaussians, n_samples).T), axis=1)

    def predict(self, X):
        """Predict the closest gaussian each sample in X belongs to.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            New data to predict.

        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the gaussian each sample belongs to.
        """
        n_samples, _ = X.shape
        tau = self._compute_tau(X, n_samples)
        tau = tau.reshape(self.n_gaussians, n_samples).T
        return np.argmax(tau, axis=1)
