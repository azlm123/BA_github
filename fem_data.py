# Translated from fem_data.jl (partial)
import numpy as np
from fem import stiffness_matrix, define_load, fix_support


def generate_random_gaussian_function(X_domain, Y_domain, num_bumps: int, max_amplitude: float, min_std_dev: float, max_std_dev: float):
    centers_x = np.random.rand(num_bumps) * X_domain
    centers_y = np.random.rand(num_bumps) * Y_domain
    amplitudes = np.random.rand(num_bumps) * max_amplitude
    std_devs = min_std_dev + np.random.rand(num_bumps) * (max_std_dev - min_std_dev)

    def f_gaussian(x, y):
        result = np.zeros_like(x, dtype=float)
        for k in range(num_bumps):
            dx = x - centers_x[k]
            dy = y - centers_y[k]
            sigma = std_devs[k]
            result += amplitudes[k] * np.exp(- (dx ** 2 + dy ** 2) / (2 * sigma ** 2))
        return result
    return f_gaussian


def generate_random_function(X, Y):
    return generate_random_gaussian_function(X, Y, 1, 1.0, 0.1 * max(X, Y), 0.5 * max(X, Y))


def generate_poisson_data(n_samples, dx, nely, nelx):
    K = stiffness_matrix(nely, nelx)
    F = np.zeros((nely + 1, nelx + 1, n_samples))
    U = np.zeros((nely + 1, nelx + 1, n_samples))
    for s in range(n_samples):
        f_s = generate_random_function(dx * nelx, dx * nely)
        F_s = f_s(np.outer(np.ones(nely + 1), dx * np.arange(nelx + 1)), np.outer(dx * np.arange(nely + 1), np.ones(nelx + 1)))
        F_RHS_s = define_load(nely, nelx, dx, f_s)
        K_fixed, F_fixed = fix_support(nely, nelx, K, F_RHS_s)
        U_s = np.linalg.solve(K_fixed.toarray(), F_fixed)
        F[:, :, s] = F_s
        U[:, :, s] = U_s.reshape((nely + 1, nelx + 1))
    return F, U


F, U = generate_poisson_data(2, 0.1, 8, 8)  # Example usage to generate data for testing
print(F[0,0,:], U[0,0,:])
