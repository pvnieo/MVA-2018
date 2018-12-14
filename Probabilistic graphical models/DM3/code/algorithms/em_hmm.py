# stdlib
from random import uniform
import numpy as np
from scipy.stats import multivariate_normal


class EmHmm:
    """Implements EM algorithm for Hidden Markov Model

    Parameters
    ----------
    trans: array, shape=[num_states, num_states]
        Transition matrix
    means: array, shape=[num_states, n_features]
        Means of the emission probabilities
    sigmas: array, shape=[num_states, n_features, n_features]
        Covariance matrices of the emission probabilities
    pi0: array, shape=[num_states]
        The initial state distribution of HMM
    max_iter : int, default: 500
        Maximum number of iterations of the EM algorithm for a single run.

    Attributes
    ----------
    _k: int
        Number of states
    _alpha: array, shape=[num_samples, sum_states]
        The alpha sequence of the HMM
    _beta: array, shape=[num_samples, sum_states]
        The beta sequence of the HMM
    log_likelihood: list(float)
        List of the values of likelihood taken over the steps of EM algo
    """

    def __init__(self, trans=4, means=None, sigmas=None, pi0=None, max_iter=500):
        self._k = trans if isinstance(trans, int) else len(trans)
        self.trans = trans
        self.means = means
        self.sigmas = sigmas
        self.pi0 = pi0
        self.max_iter = max_iter
        self.log_likelihood = [- float("inf")]
        self._alpha = None
        self._beta = None

        # log of transition matrix
        self._log_tr = None

        # log of emission observations
        self._log_obs = None

        # gamma (p(q_t = i |y))
        self._gamma = None

        # xi (p(q_t+1=i, q_t=j|y))
        self._xi = None

    def _initialisation(self, u_t):
        """Initialize trans, means, sigmas, alpha and beta sequences"""
        T, n_features = u_t.shape

        # initialize means
        if self.means is None:
            mins_feature = [np.min(u_t[:, i]) / 2 for i in range(n_features)]
            maxs_feature = [np.max(u_t[:, i]) / 2 for i in range(n_features)]
            self.means = np.array([[uniform(mins_feature[f], maxs_feature[f]) for f in range(n_features)]
                                   for i in range(self._k)])

        # initialize sigmas
        if self.sigmas is None:
            self.sigmas = np.array([np.eye(n_features) for _ in range(self._k)])

        # initialize tran
        if isinstance(self.trans, int):
            self.trans = np.random.dirichlet(np.ones(self._k), size=self._k).T

        # initialize pi0
        if self.pi0 is None:
            self.pi0 = np.random.dirichlet(np.ones(self._k), size=1).ravel()

        # initialize alpha
        self._alpha = np.zeros((T, self._k))

        # initialize beta
        self._beta = np.zeros((T, self._k))

        # initialize gamma
        self._gamma = np.zeros((T, self._k))

        # initialize xi
        self._xi = np.zeros((T - 1, self._k, self._k))

    def _alpha_recursion(self, u_t):
        """Compute the alpha recursion"""
        self._log_tr = np.log(self.trans)
        self._log_obs = np.log([multivariate_normal(self.means[i], self.sigmas[i]).pdf(u_t) for i in range(self._k)]).T

        # initialization
        self._alpha[0, :] = np.log(self.pi0) + self._log_obs[0, :]

        for i in range(1, self._alpha.shape[0]):
            term_1 = self._log_obs[i, :]
            term_2 = self._log_tr + np.tile(self._alpha[i-1, :], (self._k, 1))
            l_star = np.max(term_2, axis=1)
            term_2 = np.log(np.sum(np.exp(term_2 - l_star.reshape(-1, 1)), axis=1))
            self._alpha[i, :] = term_1 + term_2 + l_star

    def _beta_recursion(self, u_t):
        T = u_t.shape[0]
        self._beta[T-1, :] = np.log(np.ones(self._k))
        for i in range(T - 2, -1, -1):
            temp = np.tile(self._log_obs[i+1, :], (self._k, 1)) + self._log_tr.T
            temp += np.tile(self._beta[i+1, :], (self._k, 1))
            l_star = np.max(temp, axis=1)
            temp = np.log(np.sum(np.exp(temp - l_star.reshape(-1, 1)), axis=1))
            self._beta[i, :] = temp + l_star

    def _compute_loglikelihood(self):
        temp = (self._alpha + self._beta)[0, :]
        temp_star = np.max(temp)
        temp = temp_star + np.log(np.sum(np.exp(temp - temp_star)))
        self.log_likelihood.append(temp)

    def _e_step(self, u_t):
        """Compute one E-step"""
        T = u_t.shape[0]

        self._alpha_recursion(u_t)
        self._beta_recursion(u_t)

        self._compute_loglikelihood()

        # Compute gamma
        den = self._alpha + self._beta
        den_star = np.max(den, axis=1)
        den = den_star + np.log(np.sum(np.exp(den - den_star.reshape(-1, 1)), axis=1))
        self._gamma = np.exp(self._alpha + self._beta - den.reshape(-1, 1))

        # Compute xi
        for t in range(T-1):
            temp = np.tile(self._alpha[t, :], (self._k, 1))
            temp += np.tile(self._beta[t+1, :].reshape(-1, 1), (1, self._k))
            temp += self._log_tr
            temp += np.tile(self._log_obs[t+1, :].reshape(-1, 1), (1, self._k))
            self._xi[t, :, :] = np.exp(temp - den[t])

    def _m_step(self, u_t):
        """Compute one M-step"""
        d = u_t.shape[1]

        # Compute pi0
        self.pi0 = self._gamma[0, :] / np.sum(self._gamma[0, :])

        # Compute matrix of transition
        self.trans = np.sum(self._xi, axis=0)
        self.trans /= np.sum(self._xi, axis=(0, 1))

        # Compute means
        self.means = self._gamma.T.dot(u_t)
        self.means /= np.tile(np.sum(self._gamma, axis=0).reshape(-1, 1), (1, d))

        # Compute Sigmas
        s = np.sum(self._gamma, axis=0)
        for i in range(self._k):
            temp = np.sqrt(self._gamma[:, i]).reshape(-1, 1) * (u_t - self.means[i])
            self.sigmas[i, :, :] = temp.T.dot(temp) / s[i]

    def fit(self, u_t):
        """Compute EM for HMM.

        Parameters
        ----------
        u_t : array-like, shape=(n_samples, n_features)
            Training instances to EM algorithm.
        """
        # initialization
        self._initialisation(u_t)

        epsilon = 1e-10
        ite = 1

        # EM iterations
        while ite < self.max_iter:
            ite += 1
            self._e_step(u_t)
            self._m_step(u_t)
            if (self.log_likelihood[-1] - self.log_likelihood[-2]) < epsilon:
                break

    def decode(self, u_t):
        """Implement the viteri algorithm to estimate the sequence of more likely states"""
        T = u_t.shape[0]

        v = np.zeros((T, self._k))
        p = np.zeros((T, self._k))
        emission = np.array([multivariate_normal(self.means[i], self.sigmas[i]).pdf(u_t) for i in range(self._k)]).T

        # Initialization
        for k in range(self._k):
            v[0, k] = emission[0, k] * self.pi0[k]

        # Main loop
        for t in range(1, T):
            for k in range(self._k):
                temp = emission[t, k] * self.trans[:, k] * v[t-1, :]
                v[t] = self._log_obs[t, :] + np.max(self._log_tr + v[t-1, :], axis=1)
                p[t] = np.argmax(np.log(self.trans) + v[t-1, :], axis=1)

        # Retrive the path
        _map = np.zeros(T, dtype=int)
        _map[T - 1] = np.argmax(v[T - 1, :])
        for t in range(T-1, 0, -1):
            _map[t - 1] = p[t, _map[t]]

        return _map
