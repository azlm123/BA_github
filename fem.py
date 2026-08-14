# Translated from fem.jl
# Minimal Python translation using NumPy and SciPy
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix


def element_stiffness_matrix():
    k = [2/3, -1/6, -1/3, -1/6]
    KE = np.array([[k[0], k[1], k[2], k[3]],
                   [k[1], k[0], k[3], k[2]],
                   [k[2], k[3], k[0], k[1]],
                   [k[3], k[2], k[1], k[0]]], dtype=float)
    return KE


def stiffness_matrix(nely: int, nelx: int):
    KE = element_stiffness_matrix()
    n = (nelx + 1) * (nely + 1)
    K = lil_matrix((n, n), dtype=float)
    for ely in range(1, nely + 1):
        for elx in range(1, nelx + 1):
            n1 = (nely + 1) * (elx - 1) + ely
            edof = [n1 - 1 + i for i in [0, nely + 1, nely + 2, 1]]
            for i, ii in enumerate(edof):
                for j, jj in enumerate(edof):
                    K[ii, jj] += KE[i, j]
    return csr_matrix(K)


def define_load(nely: int, nelx: int, dx, f):
    F = np.zeros((nely + 1) * (nelx + 1), dtype=float)
    for ely in range(1, nely + 1):
        for elx in range(1, nelx + 1):
            n1 = (nely + 1) * (elx - 1) + ely
            edof = [n1 - 1 + i for i in [0, nely + 1, nely + 2, 1]]
            x = -dx / 2.0 + elx * dx
            y = -dx / 2.0 + ely * dx
            val = 0.25 * dx ** 2 * f(x, y)
            for idx in edof:
                F[idx] += val
    return F


def fix_support(nely: int, nelx: int, K, F):
    left = list(range(0, nely + 1))
    right = list(range((nely + 1) * nelx, (nely + 1) * (nelx + 1)))
    bottom = list(range(0, (nely + 1) * (nelx + 1), nely + 1))
    top = list(range(nely, (nely + 1) * (nelx + 1), nely + 1))
    fixed = sorted(set(left + right + bottom + top))
    K = K.tolil()
    for i in fixed:
        K[i, :] = 0.0
        K[:, i] = 0.0
        K[i, i] = 1.0
        F[i] = 0.0
    return csr_matrix(K), F


def fem(dx, nely: int, nelx: int, f_rhs):
    K = stiffness_matrix(nely, nelx)
    F = define_load(nely, nelx, dx, f_rhs)
    K, F = fix_support(nely, nelx, K, F)
    U = np.linalg.solve(K.toarray(), F)
    return U.reshape((nely + 1, nelx + 1))


if __name__ == "__main__":
    # small demo
    dx = 0.1
    nelx = 8
    nely = 8
    U = fem(dx, nely, nelx, lambda x, y: y)
    print("Solved FEM shape:", U.shape)
