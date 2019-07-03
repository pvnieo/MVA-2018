# stdlib

# 3p
import numpy as np


def levenshtein_distance(v, w):
    n, m = len(v), len(w)
    M = np.zeros((n+1, m+1))
    for i in range(n+1):
        M[i, 0] = i
    for j in range(m+1):
        M[0, j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if v[i-1] == w[j-1]:
                M[i, j] = min(M[i-1, j] + 1, M[i, j-1] + 1, M[i-1, j-1])
            else:
                M[i, j] = min(M[i-1, j] + 1, M[i, j-1] + 1, M[i-1, j-1] + 1)
    return M[n, m]


def cosine_distance(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return a.dot(b)
