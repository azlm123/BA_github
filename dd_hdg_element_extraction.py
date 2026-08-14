import numpy as np

def extract_subelements_data(U, F, nselx, nsely, nely, nelx, n_samples=None):    
    """Extract node-based subelement patches from sample arrays.

    Parameters
    - U, F: arrays with shape (nely+1, nelx+1) for single sample or
            (nely+1, nelx+1, n_samples) for multiple samples
    - nselx, nsely: number of elements per patch in x and y (must divide nelx/nely)
    - nely, nelx: total number of elements in y and x
    - n_samples: optional number of samples (last dimension). If omitted and U/F are 3D,
                 it is inferred from the data. If U/F are 2D, a single sample is assumed.

    Returns
    - F_sub, U_sub: arrays shaped
      (n_suby, n_subx, nsely+1, nselx+1, n_samples)
      where n_subx = nelx // nselx and n_suby = nely // nsely.
    """
    if U.shape[:2] != F.shape[:2]:
        raise ValueError("U and F must have the same first two dimensions")

    # Accept either single-sample 2D arrays or multi-sample 3D arrays.
    if U.ndim == 2:
        U = U[:, :, np.newaxis]
        F = F[:, :, np.newaxis]
    elif U.ndim != 3:
        raise ValueError("U and F must be 2D (single sample) or 3D (nely+1, nelx+1, n_samples)")

    n_nodes_y, n_nodes_x, ns = U.shape
    if n_nodes_y != nely + 1 or n_nodes_x != nelx + 1:
        raise ValueError("nely/nelx do not match the first two dimensions of U/F")

    # infer or validate n_samples
    if n_samples is None:
        n_samples = ns
    if ns != n_samples:
        raise ValueError("n_samples argument does not match data last dimension")

    if nelx % nselx != 0 or nely % nsely != 0:
        raise ValueError("nselx must divide nelx and nsely must divide nely")

    n_subx = nelx // nselx
    n_suby = nely // nsely
    sub_nx_nodes = nselx + 1
    sub_ny_nodes = nsely + 1

    F_sub = np.zeros((n_suby, n_subx, sub_ny_nodes, sub_nx_nodes, n_samples), dtype=F.dtype)
    U_sub = np.zeros_like(F_sub)

    for iy in range(n_suby):
        y0 = iy * nsely
        y1 = y0 + sub_ny_nodes
        for ix in range(n_subx):
            x0 = ix * nselx
            x1 = x0 + sub_nx_nodes
            F_sub[iy, ix, :, :, :] = F[y0:y1, x0:x1, :]
            U_sub[iy, ix, :, :, :] = U[y0:y1, x0:x1, :]

    return F_sub, U_sub


def reconstruct_from_subelements(F_sub, U_sub):
    """Reconstruct full-grid `F` and `U` from subelement patches.

    Parameters
    - F_sub, U_sub: arrays shaped
      (n_suby, n_subx, sub_ny_nodes, sub_nx_nodes, n_samples)
      or the same without the `n_samples` axis (4D) for single-sample.

    Returns
    - F, U: arrays shaped (nely+1, nelx+1, n_samples) (or 2D if input was 4D/single-sample)

    The function infers `nselx`, `nsely`, `nelx`, `nely` from the patch sizes.
    """
    if F_sub.shape[:2] != U_sub.shape[:2]:
        raise ValueError("F_sub and U_sub must have the same first two dimensions")

    # remember if inputs were 4D (single sample without sample axis)
    input_was_4d = (F_sub.ndim == 4)
    if F_sub.ndim == 4:
        # add sample axis
        F_sub = F_sub[:, :, :, :, np.newaxis]
        U_sub = U_sub[:, :, :, :, np.newaxis]
    if F_sub.ndim != 5:
        raise ValueError("F_sub/U_sub must be 4D (no sample) or 5D (with samples)")

    n_suby, n_subx, sub_ny_nodes, sub_nx_nodes, n_samples = F_sub.shape
    nselx = sub_nx_nodes - 1
    nsely = sub_ny_nodes - 1
    nelx = n_subx * nselx
    nely = n_suby * nsely

    F = np.zeros((nely + 1, nelx + 1, n_samples), dtype=F_sub.dtype)
    U = np.zeros_like(F)

    for iy in range(n_suby):
        y0 = iy * nsely
        y1 = y0 + sub_ny_nodes
        for ix in range(n_subx):
            x0 = ix * nselx
            x1 = x0 + sub_nx_nodes
            F[y0:y1, x0:x1, :] = F_sub[iy, ix, :, :, :]
            U[y0:y1, x0:x1, :] = U_sub[iy, ix, :, :, :]

    # If there's only one sample, return 2D arrays for convenience.
    if F.shape[2] == 1:
        return F[:, :, 0], U[:, :, 0]
    return F, U



if __name__ == "__main__":
    # small demo
    nely, nelx = 4, 4
    nselx, nsely = 2, 4
    n_samples = 2
    F = np.random.rand(nely + 1, nelx + 1, n_samples)
    U = np.random.rand(nely + 1, nelx + 1, n_samples)

    F_sub, U_sub = extract_subelements_data(U, F, nselx, nsely, nely, nelx, n_samples)
    print("Extracted subelement shapes:", F_sub.shape, U_sub.shape)

    F_recon, U_recon = reconstruct_from_subelements(F_sub, U_sub)
    print("Reconstructed shapes:", F_recon.shape, U_recon.shape)

    assert np.allclose(F_recon, F), "F reconstruction mismatch"
    assert np.allclose(U_recon, U), "U reconstruction mismatch"
    print("Reconstruction successful!")