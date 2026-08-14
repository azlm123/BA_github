# Translated from dd-hdg-util.jl (partial)
import numpy as np

il, ir, ib, it = 0, 1, 2, 3
iU, iF, idU = 0, 1, 2
EPS = 1e-8


def patchify_data(U, ny, nx):
    patches = []
    n_y, n_x, n_samples = U.shape
    step_x = nx - 1
    step_y = ny - 1
    for i in range(n_samples):
        for x in range(0, n_x - nx + 1, step_x):
            for y in range(0, n_y - ny + 1, step_y):
                patches.append(np.array(U[y:y + ny, x:x + nx, i], dtype=float))
    return np.stack(patches, axis=2) if patches else np.zeros((ny, nx, 0))


def dxdy(U, dx):
    dU_dx = np.zeros_like(U, dtype=float)
    dU_dy = np.zeros_like(U, dtype=float)
    # central differences
    dU_dx[:, 1:-1, :] = (U[:, 2:, :] - U[:, :-2, :]) / (2 * dx)
    dU_dx[:, 0, :] = (-3 * U[:, 0, :] + 4 * U[:, 1, :] - U[:, 2, :]) / (2 * dx)
    dU_dx[:, -1, :] = (3 * U[:, -1, :] - 4 * U[:, -2, :] + U[:, -3, :]) / (2 * dx)
    dU_dy[1:-1, :, :] = (U[2:, :, :] - U[:-2, :, :]) / (2 * dx)
    dU_dy[0, :, :] = (-3 * U[0, :, :] + 4 * U[1, :, :] - U[2, :, :]) / (2 * dx)
    dU_dy[-1, :, :] = (3 * U[-1, :, :] - 4 * U[-2, :, :] + U[-3, :, :]) / (2 * dx)
    return dU_dx, dU_dy


def e2f(ny: int, nx: int):
    E2F = np.zeros((ny * nx, 4), dtype=int)
    ide = 0
    for x in range(1, nx + 1):
        for y in range(1, ny + 1):
            ide += 1
            E2F[ide - 1, 0] = (x - 1) * ny + y
            E2F[ide - 1, 1] = (x - 1) * ny + y + ny
            E2F[ide - 1, 2] = (x - 1) * (ny + 1) + y
            E2F[ide - 1, 3] = (x - 1) * (ny + 1) + y + 1
    return E2F


def f2e(ny: int, nx: int):
    Fv2E = -np.ones((nx * ny + ny, 2), dtype=int)
    Fh2E = -np.ones((nx * ny + nx, 2), dtype=int)
    E2F_mat = e2f(ny, nx)
    for el in range(E2F_mat.shape[0]):
        left = E2F_mat[el, 0] - 1
        right = E2F_mat[el, 1] - 1
        bottom = E2F_mat[el, 2] - 1
        top = E2F_mat[el, 3] - 1
        Fv2E[left, 1] = el + 1
        Fv2E[right, 0] = el + 1
        Fh2E[bottom, 1] = el + 1
        Fh2E[top, 0] = el + 1
    return Fv2E, Fh2E


def normalize(field, minmax):
    return (field - minmax[0]) / (minmax[1] - minmax[0] + EPS)

