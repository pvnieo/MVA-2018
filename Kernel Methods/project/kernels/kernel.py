# 3p
import numpy as np
# project
from kernels.spectrum_kernel import DNASpectrumKernel
from kernels.mismatch_kernel import OneMismatchKernel


kernel_params = {'gaussian': ['gamma'],
                 'spectrum': ['spectrum'],
                 'mismatch': ['kgram']
                 }


class Kernel:
    def __init__(self, kernel='gaussian', kernel_params=None):
        self.kernel_name = kernel
        self.kernel_params = kernel_params

    def _kernel(self):
        '''
        Returns kernel function
        where:
            - self.kernel_name: kernel type
        returns
            - kernel function
        '''
        if self.kernel_name == 'gaussian':
            def k(x, y):
                gamma = self.kernel_params['gamma']
                return np.exp(- np.sqrt(np.linalg.norm(x - y) ** 2 / (2 * gamma ** 2)))
            return k
        elif self.kernel_name == "spectrum":
            kernel = DNASpectrumKernel(self.kernel_params['spectrum'])
            return kernel.k

    def _kernel_matrix(self, X_1, X_2):
        '''
        Computes kernel matrix
        where:
            - X_1: [n_1 x p] matrix
            - X_2: [n_2 x p] matrix
        returns:
            - K: [n_1 x n_2] kernel matrix .t. K[i, j] = k(X_1[i, :], X_2[j, :])
        '''
        n_1 = X_1.shape[0]
        n_2 = X_2.shape[0]
        K = np.zeros([n_1, n_2])
        # check if X has one dimension
        if len(X_1.shape) == 1:
            X_1 = np.expand_dims(X_1, axis=-1)
            X_2 = np.expand_dims(X_2, axis=-1)
        k = self._kernel()
        if np.array_equal(X_1, X_2):
            for i in range(n_1):
                for j in range(i+1):
                    K[i, j] = k(X_1[i, :], X_2[j, :])
                    K[j, i] = K[i, j]
        else:
            for i in range(n_1):
                for j in range(n_2):
                    K[i, j] = k(X_1[i, :], X_2[j, :])
        return K

    def compute_nedeed_matrices(self, data):
        d = {}
        for i in range(3):
            d["Ytr{}".format(i)] = data["Ytr{}".format(i)]
            d["Yval{}".format(i)] = data["Yval{}".format(i)]
        if self.kernel_name in ['gaussian']:
            for i in range(3):
                d["Ktr{}".format(i)] = self._kernel_matrix(
                    data["Xtr{}_mat100".format(i)], data["Xtr{}_mat100".format(i)])
                d["Kte{}".format(i)] = self._kernel_matrix(
                    data["Xtr{}_mat100".format(i)], data["Xte{}_mat100".format(i)])
                d["Kval{}".format(i)] = self._kernel_matrix(
                    data["Xtr{}_mat100".format(i)], data["Xval{}_mat100".format(i)])
        elif self.kernel_name == "mismatch":
            kernel = OneMismatchKernel(self.kernel_params['kgram'])
            for i in range(3):
                Xtr = kernel.embed(data["Xtr{}".format(i)])
                Xte = kernel.embed(data["Xte{}".format(i)])
                Xval = kernel.embed(data["Xval{}".format(i)])
                d["Ktr{}".format(i)] = kernel.kernel_matrix(Xtr)
                d["Kte{}".format(i)] = kernel.kernel_matrix(Xtr, Xte)
                d["Kval{}".format(i)] = kernel.kernel_matrix(Xtr, Xval)
        else:
            for i in range(3):
                d["Ktr{}".format(i)] = self._kernel_matrix(
                    data["Xtr{}".format(i)], data["Xtr{}".format(i)])
                d["Kte{}".format(i)] = self._kernel_matrix(
                    data["Xtr{}".format(i)], data["Xte{}".format(i)])
                d["Kval{}".format(i)] = self._kernel_matrix(
                    data["Xtr{}".format(i)], data["Xval{}".format(i)])
        return d
