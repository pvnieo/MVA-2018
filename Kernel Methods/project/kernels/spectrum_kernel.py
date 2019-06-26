# stdlib
from collections import defaultdict
# 3p
import numpy as np


class DNASpectrumKernel:
    def __init__(self, spectrum=3):
        self.spectrum = int(spectrum)

    def get_substrings(self, x):
        x = x[0]
        d = defaultdict(int)
        for i in range(len(x) + 1 - self.spectrum):
            d[x[i: i + self.spectrum]] += 1
        return d

    def k(self, x, y):
        dx = self.get_substrings(x)
        dy = self.get_substrings(y)
        kernel_value = 0
        for sub in dx:
            kernel_value += dx[sub] * dy[sub]
        return kernel_value
