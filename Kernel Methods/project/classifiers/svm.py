# 3p
import numpy as np
from cvxopt.solvers import qp, options
from cvxopt import matrix


class KernelSVM:
    def __init__(self, _lambda=1):
        self._lambda = _lambda
        self.alpha = None
        self.n = None

    def fit(self, K_train, y):
        K_train = K_train.astype(np.double)
        y = 2 * y - 1
        n = K_train.shape[0]
        self.n = n
        P = matrix(K_train)
        q = matrix(-y.reshape(-1, 1).astype(float))

        G = matrix(np.vstack((np.diag(y), -np.diag(y))).astype(float))
        h = matrix(np.vstack(((1/(2 * self._lambda * n)) * np.ones((n, 1)), np.zeros((n, 1)))))

        options['show_progress'] = False
        sol = qp(P, q, G, h)
        self.alpha = np.ravel(sol['x'])

    def predict(self, K_test):
        if self.alpha is None:
            raise TypeError("Model is not trained Yet!")
        if K_test.shape[0] == self.n:
            K_test = K_test.T
        prediction = np.dot(K_test, self.alpha)
        prediction = (np.sign(prediction) + 1) / 2
        return prediction
