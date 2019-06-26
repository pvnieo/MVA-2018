# 3p
import numpy as np
# project
from classifiers.svm import KernelSVM
from classifiers.logistic_regression import LogisticRegression


classifier_params = {"svm": ["_lambda"],
                     "lregression":  ["_lambda"]
                     }


class Classifier:
    def __init__(self, classifier, kernel_params=None):
        if classifier == "svm":
            self.cls = KernelSVM(**kernel_params)
        elif classifier == "lregression":
            self.cls = LogisticRegression(**kernel_params)

    def fit(self, K_train, y):
        self.cls.fit(K_train, y)

    def predict(self, K_test):
        return self.cls.predict(K_test)

    def evaluate(self, Kval, y):
        prediction = self.predict(Kval)
        return np.sum(prediction == y)/y.shape[0]
