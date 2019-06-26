# stdlib
from collections import defaultdict
from itertools import product
# 3p
import numpy as np


class OneMismatchKernel:
    def __init__(self, kgram=6):
        self.kgram = int(kgram)
        self.all_comb = self.get_possible_comb(['A', 'C', 'G', 'T'])

    def get_ngrams(self, seq):
        return list(zip(*[seq[i:] for i in range(self.kgram)]))

    def get_possible_comb(self, seq):
        return list(product(seq, repeat=self.kgram))

    def create_embed(self, seq):
        letters = ['A', 'C', 'G', 'T']
        decompose_seq = self.get_ngrams(seq)
        value = np.zeros([len(self.all_comb), ])
        for ngram in decompose_seq:
            index_ngram = self.all_comb.index(ngram)
            value[index_ngram] = value[index_ngram]+1
            copy_ngram = list(ngram)
            for ind, cur_letter in enumerate(copy_ngram):
                for letter in letters:
                    if letter != cur_letter:
                        new_ngram = list(copy_ngram)
                        new_ngram[ind] = letter
                        mismatch_ngram = tuple(new_ngram)
                        index_ngram = self.all_comb.index(mismatch_ngram)
                        value[index_ngram] = value[index_ngram] + 0.1
        return value

    def embed(self, X):
        embeded = np.empty([len(X), len(self.all_comb)])
        for i in range(len(X)):
            embeded[i, :] = self.create_embed(X[i])
        return embeded

    def kernel_matrix(self, X1, X2=[]):
        len_X2 = len(X2)
        len_X1 = len(X1)
        sim_docs_kernel_value = {}
        if len_X2 == 0:
            gram_matrix = np.zeros((len_X1, len_X1), dtype=np.float32)
            for i in range(len_X1):
                sim_docs_kernel_value[i] = np.vdot(X1[i], X1[i])

            for i in range(len_X1):
                for j in range(i, len_X1):
                    gram_matrix[i, j] = np.vdot(X1[i], X1[j]) / (sim_docs_kernel_value[i] * sim_docs_kernel_value[j]) ** 0.5
                    gram_matrix[j, i] = gram_matrix[i, j]
            # calculate Gram matrix
            return gram_matrix

        else:
            gram_matrix = np.zeros((len_X1, len_X2), dtype=np.float32)

            sim_docs_kernel_value[1] = {}
            sim_docs_kernel_value[2] = {}
            for i in range(len_X1):
                sim_docs_kernel_value[1][i] = np.vdot(X1[i], X1[i])
            for j in range(len_X2):
                sim_docs_kernel_value[2][j] = np.vdot(X1[j], X1[j])

            for i in range(len_X1):
                for j in range(len_X2):
                    gram_matrix[i, j] = np.vdot(X1[i], X2[j])/(sim_docs_kernel_value[1][i] * sim_docs_kernel_value[2][j]) ** 0.5
            return gram_matrix
