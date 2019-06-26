# 3p
import numpy as np


class LogisticRegression:
    def __init__(self, _lambda=1):
        self._lambda = _lambda
        self.alpha = None

    def _sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def sqrt_matrix(self, W):
        # To compute the square root of a symetric positive matrix
        D, V = np.linalg.eig(W)
        return np.dot(np.dot(V, np.diag(np.sqrt(D))), np.linalg.inv(V))

    def solve_WKRR(self, K, W, z, lambda_):
        n = K.shape[0]
        W_sqrt = np.real(self.sqrt_matrix(W))

        temp = np.dot(np.dot(W_sqrt, K), W_sqrt) + n * lambda_ * np.eye(n)
        return np.dot(W_sqrt, np.linalg.solve(temp, np.dot(W_sqrt, z)))

    def fit(self, K_train, y, itermax=30, eps=1e-6):
        y = 2 * y - 1
        y = y.reshape((y.shape[0], 1))
        n = K_train.shape[0]
        self.n = n
        alpha0 = np.zeros((n, 1))

        iter_ = 0
        last_alpha = 10 * alpha0 + np.ones(alpha0.shape)
        alpha = alpha0

        while (iter_ < itermax) and (np.linalg.norm(last_alpha-alpha) > eps):
            last_alpha = alpha
            m = np.dot(K_train, alpha)
            P = np.zeros((n, 1))
            W = np.zeros((n, n))
            z = np.zeros((n, 1))
            for i in range(n):
                P[i, 0] = -self._sigmoid(-y[i] * m[i])
                W[i, i] = self._sigmoid(m[i]) * self._sigmoid(-m[i])
                z[i, 0] = m[i] - (P[i, 0] * y[i])/W[i, i]
            alpha = self.solve_WKRR(K_train, W, z, self._lambda)
            iter_ = iter_ + 1
        self.alpha = alpha[:, 0]

    def predict(self, K_test):
        if self.alpha is None:
            raise TypeError("Model is not trained Yet!")
        if K_test.shape[0] == self.n:
            K_test = K_test.T
        prediction = np.dot(K_test, self.alpha)
        prediction = (np.sign(prediction) + 1) / 2
        return prediction
