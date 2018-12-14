import numpy as np


def line_search(f, df, x, dx, alpha, beta):
    t = 1
    while (np.isnan(f(x + t * dx)) or f(x + t * dx) >= (f(x) + alpha * t * df(x).T.dot(dx))) and (t > 1e-3):
        t *= beta
    return t


def newton_method(x0, f, df, d2f, eps, alpha, beta):
    x_values = [x0]
    eps = 1e-2
    x = x0
    i, max_iter = 0, 500
    while True and (i < max_iter):
        dx = np.linalg.inv(d2f(x)).dot(df(x))
        lmbda2 = df(x).T.dot(dx)
        if (0.5 * lmbda2) <= eps:
            break
        t = line_search(f, df, x, dx, alpha, beta)
        x = x - t * dx
        x_values.append(x)
        i += 1
    return x_values


def compute_objectif(v, Q, p):
    return v.T.dot(Q).dot(v) + p.T.dot(v)


def compute_f(v, Q, p, A, b, t):
    fi = A.dot(v) - b
    if (fi >= 0).any():
        return float("NaN")
    else:
        temp = t * (v.T.dot(Q).dot(v) + p.T.dot(v))
        temp -= np.sum(np.log(- fi))
        return temp


def compute_grad(v, Q, p, A, b, t):
    grad1 = t * (p + 2 * Q.dot(v))
    fi = A.dot(v) - b
    den = 1 / fi
    grad2 = A.T.dot(den)
    return grad1 + grad2


def compute_hess(v, Q, p, A, b, t):
    hess1 = t * 2 * Q
    fi = A.dot(v) - b
    den = 1 / (fi ** 2)
    diag = np.diag(den.reshape(-1))
    hess2 = A.T.dot(diag).dot(A)
    return hess1 + hess2


class SolveQP():

    def __init__(self, Q, p, A, b, mu, eps, v0=None, t0=10, alpha=0.1, beta=0.3):
        self.Q = Q
        self.p = p.reshape(-1, 1)
        self.A = A
        self.b = b.reshape(-1, 1)
        self.t_bm = t0
        self.v0 = np.zeros(Q.shape[0]).reshape(-1, 1)
        self.mu = mu
        self.eps = eps
        self.alpha = alpha
        self.beta = beta
        self.centering_v = []
        self.optimal_v = []
        self.objectif = []

    def centering_step(self, Q, p, A, b, t, v0, eps):
        def f(v):
            f = compute_f(v, Q, p, A, b, t)
            return f

        def df(v):
            df = compute_grad(v, Q, p, A, b, t)
            return df

        def d2f(v):
            d2f = compute_hess(v, Q, p, A, b, t)
            return d2f

        cent_v = newton_method(v0, f, df, d2f, eps, self.alpha, self.beta)
        self.centering_v.append(cent_v)
        return cent_v[-1]

    def barr_method(self, Q, p, A, b, v0, eps):
        def f(v):
            f = compute_objectif(v, Q, p)
            return f

        m = A.shape[0]
        v = v0
        self.optimal_v.append(v)
        while (m / self.t_bm) >= eps:
            v = self.centering_step(Q, p, A, b, self.t_bm, v, eps)
            self.optimal_v.append(v)
            self.objectif.append(f(v))
            self.t_bm *= self.mu
        
    def fit(self):
        self.barr_method(self.Q, self.p, self.A, self.b, self.v0, self.eps)
