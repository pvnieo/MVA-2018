import numpy as np
from scipy.special import expit


class LogisticRegression:
    """Classifier based on the logistic regression model.

    This class implements the logistic regression model learned on data X and Y where:
        - X € IR²
        - Y € {0, 1}
    """

    def __init__(self, W=(0, 0, 0)):
        self.name = "logistic regression"
        self.W = np.array(W)

    def one_pass_IRLS(self, W, X_p, Y):
        """Compute one iteration of the IRLS algorithm (Newton-Raphson)
        """
        nu = expit(np.dot(X_p, W))
        nabla_l = X_p.T.dot(Y - nu)
        Hl = X_p.T.dot(np.diag(nu * (1 - nu)).dot(X_p))
        Hl_1 = np.linalg.pinv(Hl)
        W = W + Hl_1.dot(nabla_l)
        return W

    def IRLS(self, X_p, Y, iter=20):
        """Compute iter iteration of the IRLS algorithm
           Newton methods does not require lot of iteration to converge, pratically: iterations of transition phase + 5
        """
        counter = 0
        W = np.array((0, 0, 0))
        X = np.delete(X_p, -1, 1)
        while counter < iter and self.score(X, Y) < 1:
            W = self.one_pass_IRLS(W, X_p, Y)
            counter += 1
        return W

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
        self.W = self.IRLS(X_p, Y)

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
        C = expit(np.dot(X_p, self.W))
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

        def line(x): return (- self.W[2] - self.W[0] * x) / self.W[1]
        return (x, line(x))
