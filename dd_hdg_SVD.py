import numpy as np
import os
import matplotlib.pyplot as plt

from dd_hdg_element_extraction import *
from dd_hdg_face_operations import *
from fem_data import *

# Configuration Parameters
nely, nelx = 64, 64
nselx, nsely = 8, 8

# Total sample allocation (1000 total)
n_train_raw = 160
n_val = 20
n_test = 20
n_samples_total = n_train_raw + n_val + n_test  # 200 samples
choose_k = True  # If True, manually set truncation ranks; if False, use energy-based selection

dx = 0.1
ENERGY_THRESHOLD = 0.999


# =============================================================================
# 1. FLATTENING & METADATA UTILITIES
# =============================================================================

def flatten_subelements_to_rows(U_sub, keep_boundary=False):
    """Flatten each sub-element snapshot into a single row.
    
    Metadata Columns: [row_index, sample_index, nx_index, ny_index, is_boundary]
    """
    n_suby, n_subx, ny_nodes, nx_nodes, n_samples = U_sub.shape
    if ny_nodes < 3 or nx_nodes < 3:
        raise ValueError("U_sub patches must have at least 3x3 nodes")
    
    if keep_boundary:
        interior = U_sub
        feature_dim = ny_nodes * nx_nodes
    else:
        interior = U_sub[:, :, 1:-1, 1:-1, :]
        feature_dim = (ny_nodes - 2) * (nx_nodes - 2)

    matrix = interior.transpose(0, 1, 4, 2, 3).reshape(n_suby * n_subx * n_samples, feature_dim)
    
    row_index = np.arange(n_suby * n_subx * n_samples, dtype=np.int64)
    sample_index = np.tile(np.arange(n_samples, dtype=np.int64), n_suby * n_subx)
    nx_index = np.tile(np.repeat(np.arange(n_subx, dtype=np.int64), n_samples), n_suby)
    ny_index = np.repeat(np.arange(n_suby, dtype=np.int64), n_subx * n_samples)
    
    is_boundary = (
        (nx_index == 0) | (nx_index == n_subx - 1) |
        (ny_index == 0) | (ny_index == n_suby - 1)
    ).astype(np.int64)

    row_metadata = np.column_stack((row_index, sample_index, nx_index, ny_index, is_boundary))
    return matrix, row_metadata


def flatten_face_dict_to_matrix(face_dict, n_subx, n_suby, n_samples):
    """Flatten all face values into a single 2D matrix.
    
    Metadata Columns: [row_index, sample_index, nx_index, ny_index, face_kind, is_global_boundary]
    """
    ordered_faces = [face_dict["bottom"], face_dict["right"], face_dict["top"], face_dict["left"]]
    row_blocks = []
    meta_blocks = []

    for face_kind, face_values in enumerate(ordered_faces):
        _, _, face_length, _ = face_values.shape
        row_blocks.append(face_values.transpose(0, 1, 3, 2).reshape(n_suby * n_subx * n_samples, face_length))
        
        row_index = np.arange(n_suby * n_subx * n_samples, dtype=np.int64)
        sample_index = np.tile(np.arange(n_samples, dtype=np.int64), n_suby * n_subx)
        nx_index = np.tile(np.repeat(np.arange(n_subx, dtype=np.int64), n_samples), n_suby)
        ny_index = np.repeat(np.arange(n_suby, dtype=np.int64), n_subx * n_samples)
        face_index = np.full(n_suby * n_subx * n_samples, face_kind, dtype=np.int64)
        
        if face_kind == 0:    # Bottom face
            is_global_boundary = (ny_index == 0)
        elif face_kind == 1:  # Right face
            is_global_boundary = (nx_index == n_subx - 1)
        elif face_kind == 2:  # Top face
            is_global_boundary = (ny_index == n_suby - 1)
        elif face_kind == 3:  # Left face
            is_global_boundary = (nx_index == 0)
            
        is_global_boundary = is_global_boundary.astype(np.int64)
        meta_blocks.append(
            np.column_stack((row_index, sample_index, nx_index, ny_index, face_index, is_global_boundary))
        )
        
    return np.vstack(row_blocks), np.vstack(meta_blocks)


def flatten_corner_dict_to_matrix(corner_dict, n_subx, n_suby, n_samples):
    """Flatten corner values into a single 2D matrix.
    
    Metadata Columns: [row_index, sample_index, nx_index, ny_index, corner_kind, is_global_boundary]
    """
    ordered_corners = [corner_dict["bottom_left"], corner_dict["bottom_right"], 
                       corner_dict["top_right"], corner_dict["top_left"]]
    row_blocks = []
    meta_blocks = []

    for corner_kind, corner_values in enumerate(ordered_corners):
        row_blocks.append(corner_values.reshape(n_suby * n_subx * n_samples, 1))
        
        row_index = np.arange(n_suby * n_subx * n_samples, dtype=np.int64)
        sample_index = np.tile(np.arange(n_samples, dtype=np.int64), n_suby * n_subx)
        nx_index = np.tile(np.repeat(np.arange(n_subx, dtype=np.int64), n_samples), n_suby)
        ny_index = np.repeat(np.arange(n_suby, dtype=np.int64), n_subx * n_samples)
        corner_index = np.full(n_suby * n_subx * n_samples, corner_kind, dtype=np.int64)
        
        if corner_kind == 0:    # Bottom-left
            is_global_boundary = (nx_index == 0) | (ny_index == 0)
        elif corner_kind == 1:  # Bottom-right
            is_global_boundary = (nx_index == n_subx - 1) | (ny_index == 0)
        elif corner_kind == 2:  # Top-right
            is_global_boundary = (nx_index == n_subx - 1) | (ny_index == n_suby - 1)
        elif corner_kind == 3:  # Top-left
            is_global_boundary = (nx_index == 0) | (ny_index == n_suby - 1)
            
        is_global_boundary = is_global_boundary.astype(np.int64)
        meta_blocks.append(
            np.column_stack((row_index, sample_index, nx_index, ny_index, corner_index, is_global_boundary))
        )
        
    return np.vstack(row_blocks), np.vstack(meta_blocks)

def modify_face_fluxes_with_corner_gradient(J_face, U_sub, dx, dy):
    n_suby, n_subx, rn, cn, n_samples = U_sub.shape
    
    delta_x = (cn - 1) * float(dx)
    delta_y = (rn - 1) * float(dy)

    u_bl = U_sub[:, :, 0, 0, :]       
    u_br = U_sub[:, :, 0, cn - 1, :]  
    u_tl = U_sub[:, :, rn - 1, 0, :]  
    u_tr = U_sub[:, :, rn - 1, cn - 1, :] 

    J_face_mod = {}

    # Ensure shape is (n_suby, n_subx, 1, n_samples) for broadcasting across face_length
    grad_bottom = (-(u_br - u_bl) / delta_x)[:, :, np.newaxis, :]
    grad_right  = ((u_tr - u_br) / delta_y)[:, :, np.newaxis, :]
    grad_top    = ((u_tr - u_tl) / delta_x)[:, :, np.newaxis, :]
    grad_left   = (-(u_tl - u_bl) / delta_y)[:, :, np.newaxis, :]

    J_face_mod["bottom"] = J_face["bottom"] - grad_bottom
    J_face_mod["right"]  = J_face["right"]  - grad_right
    J_face_mod["top"]    = J_face["top"]    - grad_top
    J_face_mod["left"]   = J_face["left"]   - grad_left

    return J_face_mod
# =============================================================================
# 2. SVD & REDUCED-ORDER MODELING (ROM) UTILITIES
# =============================================================================

def apply_svd_to_matrix(M, k=None):
    """Compute SVD for matrix M (handles zero-variance/constant matrices safely)."""
    if np.allclose(M, 0):
        n_rows, n_cols = M.shape
        rank = k if k is not None else min(n_rows, n_cols)
        return np.zeros((n_rows, rank)), np.zeros(rank), np.zeros((rank, n_cols))

    U_mat, s, Vt = np.linalg.svd(M, full_matrices=False)
    if k is None:
        return U_mat, s, Vt
    return U_mat[:, :k], s[:k], Vt[:k, :]


def determine_best_k_from_s(s, energy_threshold=0.999, max_k=None):
    """Estimate truncation rank k based on cumulative energy."""
    if np.allclose(s, 0):
        return 1, 1, np.array([1.0])

    energies = s ** 2
    cum = np.cumsum(energies)
    total = cum[-1]
    frac = cum / (total + 1e-15)
    
    energy_k = int(np.searchsorted(frac, energy_threshold)) + 1
    if max_k is not None:
        energy_k = min(energy_k, int(max_k))

    log_s = np.log(s + 1e-16)
    if log_s.size >= 3:
        search_limit = min(
            log_s.size,
            max(3, max(energy_k + 8, int(np.ceil(energy_k * 4))))
        )
        search_log_s = log_s[:search_limit]
        diff2 = np.diff(search_log_s, n=2)
        elbow_k = int(np.argmax(-diff2)) + 2
        elbow_k = min(max(elbow_k, 1), search_limit)
    else:
        elbow_k = energy_k

    return energy_k, elbow_k, frac


def project_to_rom(full_matrix, U_basis, mean=None):
    """Project snapshots onto the POD/SVD basis."""
    centered_matrix = full_matrix if mean is None else (full_matrix - mean)
    return centered_matrix @ U_basis


def reconstruct_from_rom(rom_coeffs, basis_matrix, mean_vector=None):
    """Reconstruct full physical field from latent ROM coefficients and POD basis."""
    reconstructed = rom_coeffs @ basis_matrix.T
    if mean_vector is not None:
        reconstructed = reconstructed + mean_vector
    return reconstructed

def plot_frac_curve(frac, title, fig_out):
    """Plot cumulative energy fraction profile."""
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(1, frac.size + 1), frac)
    plt.axhline(0.999, color='gray', linestyle='--', label='0.999 Energy Threshold')
    plt.xlabel('Number of POD Modes (k)')
    plt.ylabel('Cumulative Energy Fraction')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_out, dpi=200)
    plt.close()

def compute_reconstruction_errors_vs_k(
    test_matrix,
    full_Vt,
    mean_vector,
    max_k=None,
    field_name="Field",
    plot_fig=True,
    fig_out="reconstruction_error_vs_k.png"
):
    """Trace and plot relative Frobenius norm error of test data reconstruction vs. rank k.
    
    Relative Error(k) = || M_test - ( (M_test - mean) @ V_k @ V_k^T + mean ) ||_F / || M_test ||_F
    
    Parameters
    ----------
    test_matrix : ndarray
        Original test data matrix of shape (n_test_samples, n_features).
    full_Vt : ndarray
        Full right singular vectors from training SVD of shape (rank_max, n_features).
    mean_vector : ndarray
        Mean vector computed from training data of shape (1, n_features).
    max_k : int, optional
        Maximum rank k to evaluate. If None, uses total available rows in full_Vt.
    field_name : str, optional
        Label used in output logs and plots.
    plot_fig : bool, optional
        Whether to generate and save a matplotlib plot.
    fig_out : str, optional
        Path where the plot PNG will be saved.

    Returns
    -------
    rel_errors : ndarray
        1D array containing relative reconstruction errors for k = 1, ..., max_k.
    k_values : ndarray
        1D array containing k values from 1 to max_k.
    """
    total_modes = full_Vt.shape[0]
    if max_k is None or max_k > total_modes:
        max_k = total_modes

    k_values = np.arange(1, max_k + 1)
    rel_errors = np.zeros(max_k, dtype=np.float64)

    test_norm = np.linalg.norm(test_matrix, ord='fro')+1e-15  # Avoid division by zero
    if np.isclose(test_norm, 0.0):
        print(f"[{field_name}] Warning: Test matrix norm is zero. Returning zero errors.")
        return rel_errors, k_values

    centered_test = test_matrix - mean_vector

    for idx, k in enumerate(k_values):
        basis_k = full_Vt[:k, :].T  # Shape: (n_features, k)
        rom_coeffs_k = centered_test @ basis_k  # Shape: (n_test_samples, k)
        reconstructed_k = (rom_coeffs_k @ basis_k.T) + mean_vector
        
        error_norm = np.linalg.norm(test_matrix - reconstructed_k, ord='fro')
        rel_errors[idx] = error_norm / test_norm

    if plot_fig:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 1. Linear Scale Subplot (Real Values)
        axes[0].plot(k_values, rel_errors, marker='o', markersize=3, linewidth=1.5, color='tab:blue')
        axes[0].set_xlabel('Truncation Rank (k)')
        axes[0].set_ylabel('Relative Reconstruction Error')
        axes[0].set_title(f'{field_name} Error (Linear Scale)')
        axes[0].grid(True, linestyle="--", alpha=0.6)

        # 2. Logarithmic Scale Subplot
        axes[1].semilogy(k_values, rel_errors, marker='o', markersize=3, linewidth=1.5, color='tab:red')
        axes[1].set_xlabel('Truncation Rank (k)')
        axes[1].set_ylabel('Relative Reconstruction Error (Log Scale)')
        axes[1].set_title(f'{field_name} Error (Log Scale)')
        axes[1].grid(True, which="both", linestyle="--", alpha=0.6)

        plt.suptitle(f'Test Data Reconstruction Error vs. k ({field_name})', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(fig_out, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  -> Saved reconstruction error trace plot to '{fig_out}'")

    return rel_errors, k_values
# =============================================================================
# 3. DATA EXTRACTION AND SPLITTING HELPER
# =============================================================================

def process_and_flatten_dataset(U_data, F_data, nselx, nsely, nely, nelx, dx):
    """Extract sub-elements, compute fluxes, and flatten matrices for a dataset split."""
    n_samples = U_data.shape[2]  # Assuming shape is (ny, nx, n_samples)
    
    F_sub, U_sub = extract_subelements_data(U_data, F_data, nselx=nselx, nsely=nsely, nely=nely, nelx=nelx, n_samples=n_samples)
    
    J_face = compute_outgoing_fluxes_with_corners(U_sub, dx, interior_method="central", boundary_method="inside_second")
    err_before = np.abs(J_face["top"][0, 0, :, :] + J_face["bottom"][1, 0, :, :]).max()
    J_face = modify_face_fluxes_with_corner_gradient(J_face, U_sub, dx, dx)
    # Check continuity between Top face of (ny, nx) and Bottom face of (ny+1, nx)
    
    err_after  = np.abs(J_face["top"][0, 0, :, :] + J_face["bottom"][1, 0, :, :]).max()

    print(f"Max Interface Flux Error - Before: {err_before:.2e} | After: {err_after:.2e}")

    U_sub_mat, U_sub_meta = flatten_subelements_to_rows(U_sub, keep_boundary=True)
    F_sub_mat, F_sub_meta = flatten_subelements_to_rows(F_sub, keep_boundary=True)

    modified_U_sub = modify_subdomain_boundary_values(U_sub, dx, dx)
    U_face = extract_boundary_values(modified_U_sub)
    U_corners = extract_corner_values(U_sub)

    U_corners_mat, U_corners_meta = flatten_corner_dict_to_matrix(U_corners, nselx, nsely, n_samples)
    U_face_mat, U_face_meta = flatten_face_dict_to_matrix(U_face, nselx, nsely, n_samples)
    J_face_mat, J_face_meta = flatten_face_dict_to_matrix(J_face, nselx, nsely, n_samples)

    return {
        "U_sub_mat": U_sub_mat, "U_sub_meta": U_sub_meta,
        "F_sub_mat": F_sub_mat, "F_sub_meta": F_sub_meta,
        "U_face_mat": U_face_mat, "U_face_meta": U_face_meta,
        "J_face_mat": J_face_mat, "J_face_meta": J_face_meta,
        "U_corners_mat": U_corners_mat, "U_corners_meta": U_corners_meta
    }
def augment_poisson_dataset_d4(F_all, U_all):
    """Enrich 2D spatial fields using the D4 dihedral group symmetry.
    
    Applies 4 rotations (0, 90, 180, 270 deg) and 2 axial reflections (x and y).
    Expects input shape: (ny, nx, n_samples)
    Returns augmented arrays of shape: (ny, nx, 8 * n_samples)
    """
    F_list, U_list = [], []

    for i in range(F_all.shape[2]):
        f_orig = F_all[:, :, i]
        u_orig = U_all[:, :, i]

        # 4 Rotations for non-reflected state
        for k in range(4):
            F_list.append(np.rot90(f_orig, k=k, axes=(0, 1)))
            U_list.append(np.rot90(u_orig, k=k, axes=(0, 1)))

        # 4 Rotations for horizontally reflected state (x-reflection)
        f_flip_x = np.fliplr(f_orig)
        u_flip_x = np.fliplr(u_orig)
        for k in range(4):
            F_list.append(np.rot90(f_flip_x, k=k, axes=(0, 1)))
            U_list.append(np.rot90(u_flip_x, k=k, axes=(0, 1)))

    F_aug = np.stack(F_list, axis=2)
    U_aug = np.stack(U_list, axis=2)

    return F_aug, U_aug

# =============================================================================
# 4. MAIN WORKFLOW
# =============================================================================

def main():
    print(f"Step 1: Generating {n_samples_total} base Poisson solution snapshots...")
    F_raw, U_raw = generate_poisson_data(n_samples_total, dx, nely, nelx)

    print("Step 1b: Applying D4 symmetry augmentation (8x enrichment)...")
    F_all, U_all = augment_poisson_dataset_d4(F_raw, U_raw)
    
    # Update total sample count to reflect the 8x augmentation
    n_samples_augmented = F_all.shape[2]
    print(f"  -> Total augmented snapshots: {n_samples_augmented}")

    # Scale split sizes accordingly (e.g., maintaining an 80/10/10 ratio)
    n_train_aug = n_train_raw * 8
    n_val_aug = n_val * 8
    n_test_aug = n_test * 8

    print(f"Step 2: Partitioning dataset into splits...")
    print(f"  -> Train Split      : {n_train_aug} samples")
    print(f"  -> Validation Split : {n_val_aug} samples")
    print(f"  -> Test Split       : {n_test_aug} samples")

    # Slice raw PDE solution fields along the sample dimension (axis=2)
    U_train_raw = U_all[:, :, :n_train_aug]
    F_train_raw = F_all[:, :, :n_train_aug]

    U_val = U_all[:, :, n_train_aug:n_train_aug + n_val_aug]
    F_val = F_all[:, :, n_train_aug:n_train_aug + n_val_aug]

    U_test = U_all[:, :, n_train_aug + n_val_aug:]
    F_test = F_all[:, :, n_train_aug + n_val_aug:]

    print("Step 3: Extracting sub-elements and matrices for each split...")
    train_data = process_and_flatten_dataset(U_train_raw, F_train_raw, nselx, nsely, nely, nelx, dx)
    val_data = process_and_flatten_dataset(U_val, F_val, nselx, nsely, nely, nelx, dx)
    test_data = process_and_flatten_dataset(U_test, F_test, nselx, nsely, nely, nelx, dx)

    # -------------------------------------------------------------------------
    # STEP 4: COMPUTE SVD BASES EXCLUSIVELY ON THE TRAINING SPLIT
    # -------------------------------------------------------------------------
    print("Step 4: Computing SVD/POD bases EXCLUSIVELY on Training data...")
    
    # -------------------------------------------------------------------------
    # STEP 4: COMPUTE SVD BASES EXCLUSIVELY ON THE TRAINING SPLIT
    # -------------------------------------------------------------------------
    print("Step 4: Computing SVD/POD bases EXCLUSIVELY on Training data...")
    
    tr_meta_sub = train_data["U_sub_meta"]
    tr_meta_face = train_data["U_face_meta"]
    tr_meta_corner = train_data["U_corners_meta"]

    # --- Parent Masks for Volume Fields ---
    is_bnd_sub_tr = tr_meta_sub[:, 4] == 1
    is_int_sub_tr = ~is_bnd_sub_tr

    # --- Geometric Masks for SVD Math ---
    is_global_bnd_face_tr = tr_meta_face[:, 5] == 1
    is_global_int_face_tr = ~is_global_bnd_face_tr

    is_global_bnd_corner_tr = tr_meta_corner[:, 5] == 1
    is_global_int_corner_tr = ~is_global_bnd_corner_tr

    # --- A. INTERNAL SVD BASES ---
    U_int_mat_tr = train_data["U_sub_mat"][is_int_sub_tr]
    U_int_mean = U_int_mat_tr.mean(axis=0, keepdims=True)
    _, U_int_s, U_int_Vt = apply_svd_to_matrix(U_int_mat_tr - U_int_mean)
    U_int_k, U_int_k_elbow, U_int_frac = determine_best_k_from_s(U_int_s, energy_threshold=ENERGY_THRESHOLD)

    F_int_mat_tr = train_data["F_sub_mat"][is_int_sub_tr]
    F_int_mean = F_int_mat_tr.mean(axis=0, keepdims=True)
    _, F_int_s, F_int_Vt = apply_svd_to_matrix(F_int_mat_tr - F_int_mean)
    F_int_k, F_int_k_elbow, _ = determine_best_k_from_s(F_int_s, energy_threshold=ENERGY_THRESHOLD)

    U_face_int_mat_tr = train_data["U_face_mat"][is_global_int_face_tr]
    U_face_int_mean = U_face_int_mat_tr.mean(axis=0, keepdims=True)
    _, U_face_int_s, U_face_int_Vt = apply_svd_to_matrix(U_face_int_mat_tr - U_face_int_mean)
    U_face_int_k, U_face_int_k_elbow, U_face_int_frac = determine_best_k_from_s(U_face_int_s, energy_threshold=ENERGY_THRESHOLD)

    J_face_int_mat_tr = train_data["J_face_mat"][is_global_int_face_tr]
    J_face_int_mean = J_face_int_mat_tr.mean(axis=0, keepdims=True)
    _, J_face_int_s, J_face_int_Vt = apply_svd_to_matrix(J_face_int_mat_tr - J_face_int_mean)
    J_face_int_k, J_face_int_k_elbow, J_face_int_frac = determine_best_k_from_s(J_face_int_s, energy_threshold=ENERGY_THRESHOLD)

    U_corners_int_mat_tr = train_data["U_corners_mat"][is_global_int_corner_tr]
    U_corners_int_mean = U_corners_int_mat_tr.mean(axis=0, keepdims=True)

    # --- B. BOUNDARY SVD BASES ---
    U_bnd_mat_tr = train_data["U_sub_mat"][is_bnd_sub_tr]
    U_bnd_mean = U_bnd_mat_tr.mean(axis=0, keepdims=True)
    _, U_bnd_s, U_bnd_Vt = apply_svd_to_matrix(U_bnd_mat_tr - U_bnd_mean)
    U_bnd_k, U_bnd_k_elbow, U_bnd_frac = determine_best_k_from_s(U_bnd_s, energy_threshold=ENERGY_THRESHOLD)

    F_bnd_mat_tr = train_data["F_sub_mat"][is_bnd_sub_tr]
    F_bnd_mean = F_bnd_mat_tr.mean(axis=0, keepdims=True)
    _, F_bnd_s, F_bnd_Vt = apply_svd_to_matrix(F_bnd_mat_tr - F_bnd_mean)
    F_bnd_k, F_bnd_k_elbow, _ = determine_best_k_from_s(F_bnd_s, energy_threshold=ENERGY_THRESHOLD)

    U_face_bnd_mat_tr = train_data["U_face_mat"][is_global_bnd_face_tr]
    U_face_bnd_mean = U_face_bnd_mat_tr.mean(axis=0, keepdims=True)
    _, U_face_bnd_s, U_face_bnd_Vt = apply_svd_to_matrix(U_face_bnd_mat_tr - U_face_bnd_mean)
    U_face_bnd_k, U_face_bnd_k_elbow, _ = determine_best_k_from_s(U_face_bnd_s, energy_threshold=ENERGY_THRESHOLD)

    J_face_bnd_mat_tr = train_data["J_face_mat"][is_global_bnd_face_tr]
    J_face_bnd_mean = J_face_bnd_mat_tr.mean(axis=0, keepdims=True)
    _, J_face_bnd_s, J_face_bnd_Vt = apply_svd_to_matrix(J_face_bnd_mat_tr - J_face_bnd_mean)
    J_face_bnd_k, J_face_bnd_k_elbow, _ = determine_best_k_from_s(J_face_bnd_s, energy_threshold=ENERGY_THRESHOLD)

    U_corners_bnd_mat_tr = train_data["U_corners_mat"][is_global_bnd_corner_tr]
    U_corners_bnd_mean = U_corners_bnd_mat_tr.mean(axis=0, keepdims=True)

    # --- Rank Truncation ---
    U_shared_k = min(max(U_int_k, U_bnd_k, U_int_k_elbow, U_bnd_k_elbow), U_int_Vt.shape[0], U_bnd_Vt.shape[0])
    print(f"  -> Truncation Ranks: U_int_k={U_int_k}, U_bnd_k={U_bnd_k}, U_int_k_elbow={U_int_k_elbow}, U_bnd_k_elbow={U_bnd_k_elbow}")
    F_shared_k = min(max(F_int_k, F_bnd_k, F_int_k_elbow, F_bnd_k_elbow), F_int_Vt.shape[0], F_bnd_Vt.shape[0])
    print(f"  -> Truncation Ranks: F_int_k={F_int_k}, F_bnd_k={F_bnd_k}, F_int_k_elbow={F_int_k_elbow}, F_bnd_k_elbow={F_bnd_k_elbow}")
    U_face_shared_k = min(max(U_face_int_k, U_face_bnd_k, U_face_int_k_elbow, U_face_bnd_k_elbow), U_face_int_Vt.shape[0], U_face_bnd_Vt.shape[0])
    print(f"  -> Truncation Ranks: U_face_int_k={U_face_int_k}, U_face_bnd_k={U_face_bnd_k}, U_face_int_k_elbow={U_face_int_k_elbow}, U_face_bnd_k_elbow={U_face_bnd_k_elbow}")
    J_face_shared_k = min(max(J_face_int_k, J_face_bnd_k, J_face_int_k_elbow, J_face_bnd_k_elbow), J_face_int_Vt.shape[0], J_face_bnd_Vt.shape[0])
    print(f"  -> Truncation Ranks: J_face_int_k={J_face_int_k}, J_face_bnd_k={J_face_bnd_k}, J_face_int_k_elbow={J_face_int_k_elbow}, J_face_bnd_k_elbow={J_face_bnd_k_elbow}")
    if choose_k==True:
        U_shared_k = 6
        F_shared_k = 6
        U_face_shared_k = 3
        J_face_shared_k = 3

    print(f"  -> Truncation Ranks: U_sub={U_shared_k}, F_sub={F_shared_k}, U_face={U_face_shared_k}, J_face={J_face_shared_k}")
    # -------------------------------------------------------------------------
    # STEP 4b: TRACE RELATIVE RECONSTRUCTION ERROR ON TEST DATA VS. k
    # -------------------------------------------------------------------------
    print("\nStep 4b: Computing Test Data Reconstruction Errors vs. Rank k for all fields...")
    
    test_meta_sub = test_data["U_sub_meta"]
    test_meta_face = test_data["U_face_meta"]

    is_int_sub_ts = test_meta_sub[:, 4] == 0
    is_int_face_ts = test_meta_face[:, 5] == 0

    U_int_mat_ts = test_data["U_sub_mat"][is_int_sub_ts]
    F_int_mat_ts = test_data["F_sub_mat"][is_int_sub_ts]
    U_face_int_mat_ts = test_data["U_face_mat"][is_int_face_ts]
    J_face_int_mat_ts = test_data["J_face_mat"][is_int_face_ts]
    
    # Trace 1: U_sub (Interior)
    err_U_sub_int, _ = compute_reconstruction_errors_vs_k(
        test_matrix=U_int_mat_ts,
        full_Vt=U_int_Vt,
        mean_vector=U_int_mean,
        max_k=None,
        field_name="U_sub (Interior)",
        fig_out="rel_error_U_sub_int_test.png"
    )

    # Trace 2: F_sub (Interior)
    err_F_sub_int, _ = compute_reconstruction_errors_vs_k(
        test_matrix=F_int_mat_ts,
        full_Vt=F_int_Vt,
        mean_vector=F_int_mean,
        max_k=None,
        field_name="F_sub (Interior)",
        fig_out="rel_error_F_sub_int_test.png"
    )

    # Trace 3: U_face (Interior)
    err_U_face_int, _ = compute_reconstruction_errors_vs_k(
        test_matrix=U_face_int_mat_ts,
        full_Vt=U_face_int_Vt,
        mean_vector=U_face_int_mean,
        max_k=None,
        field_name="U_face (Interior)",
        fig_out="rel_error_U_face_int_test.png"
    )

    # Trace 4: J_face (Interior)
    err_J_face_int, _ = compute_reconstruction_errors_vs_k(
        test_matrix=J_face_int_mat_ts,
        full_Vt=J_face_int_Vt,
        mean_vector=J_face_int_mean,
        max_k=None,
        field_name="J_face (Interior)",
        fig_out="rel_error_J_face_int_test.png"
    )
    err_U_sub_bnd, _ = compute_reconstruction_errors_vs_k(
        test_matrix=test_data["U_sub_mat"][~is_int_sub_ts],
        full_Vt=U_bnd_Vt,
        mean_vector=U_bnd_mean,
        max_k=None,
        field_name="U_sub (Boundary)",
        fig_out="rel_error_U_sub_bnd_test.png"
    )
    err_F_sub_bnd, _ = compute_reconstruction_errors_vs_k(
        test_matrix=test_data["F_sub_mat"][~is_int_sub_ts],
        full_Vt=F_bnd_Vt,
        mean_vector=F_bnd_mean,
        max_k=None,
        field_name="F_sub (Boundary)",
        fig_out="rel_error_F_sub_bnd_test.png"
    )
    err_U_face_bnd, _ = compute_reconstruction_errors_vs_k(
        test_matrix=test_data["U_face_mat"][~is_int_face_ts],
        full_Vt=U_face_bnd_Vt,
        mean_vector=U_face_bnd_mean,
        max_k=None,
        field_name="U_face (Boundary)",
        fig_out="rel_error_U_face_bnd_test.png"
    )

    err_J_face_bnd, _ = compute_reconstruction_errors_vs_k(
        test_matrix=test_data["J_face_mat"][~is_int_face_ts],
        full_Vt=J_face_bnd_Vt,
        mean_vector=J_face_bnd_mean,
        max_k=None,
        field_name="J_face (Boundary)",
        fig_out="rel_error_J_face_bnd_test.png"
    )




    print(f"  -> Final Test Error @ k={U_shared_k} [U_sub] : {err_U_sub_int[U_shared_k-1]:.4e} {err_U_sub_bnd[U_shared_k-1]:.4e}")
    print(f"  -> Final Test Error @ k={F_shared_k} [F_sub] : {err_F_sub_int[F_shared_k-1]:.4e} {err_F_sub_bnd[F_shared_k-1]:.4e}")
    print(f"  -> Final Test Error @ k={U_face_shared_k} [U_face]: {err_U_face_int[U_face_shared_k-1]:.4e} {err_U_face_bnd[U_face_shared_k-1]:.4e}")
    print(f"  -> Final Test Error @ k={J_face_shared_k} [J_face]: {err_J_face_int[J_face_shared_k-1]:.4e} {err_J_face_bnd[J_face_shared_k-1]:.4e}\n")


    U_int_basis = U_int_Vt.T[:, :U_shared_k]
    U_bnd_basis = U_bnd_Vt.T[:, :U_shared_k]
    F_int_basis = F_int_Vt.T[:, :F_shared_k]
    F_bnd_basis = F_bnd_Vt.T[:, :F_shared_k]
    U_face_int_basis = U_face_int_Vt.T[:, :U_face_shared_k]
    U_face_bnd_basis = U_face_bnd_Vt.T[:, :U_face_shared_k]
    J_face_int_basis = J_face_int_Vt.T[:, :J_face_shared_k]
    J_face_bnd_basis = J_face_bnd_Vt.T[:, :J_face_shared_k]

    # --- SAVE ALL BASES TO A DEDICATED FILE ---
    np.savez_compressed(
        "hdg_rom_bases_cornerflux.npz",
        U_sub_int_basis=U_int_basis, U_sub_bnd_basis=U_bnd_basis,
        F_sub_int_basis=F_int_basis, F_sub_bnd_basis=F_bnd_basis,
        U_face_int_basis=U_face_int_basis, U_face_bnd_basis=U_face_bnd_basis,
        J_face_int_basis=J_face_int_basis, J_face_bnd_basis=J_face_bnd_basis,
        U_sub_int_mean=U_int_mean, U_sub_bnd_mean=U_bnd_mean,
        F_sub_int_mean=F_int_mean, F_sub_bnd_mean=F_bnd_mean,
        U_face_int_mean=U_face_int_mean, U_face_bnd_mean=U_face_bnd_mean,
        J_face_int_mean=J_face_int_mean, J_face_bnd_mean=J_face_bnd_mean,
        U_corners_int_mean=U_corners_int_mean, U_corners_bnd_mean=U_corners_bnd_mean
    )
    print("  -> Saved all SVD bases and means to 'hdg_rom_bases_cornerflux.npz'")

    # -------------------------------------------------------------------------
    # STEP 5: PROJECT ALL SPLITS (TRAIN/VAL/TEST) ONTO TRAINED BASES
    # -------------------------------------------------------------------------
    print("Step 5: Projecting Train, Val, and Test datasets onto Training POD bases...")


    splits = {"train": train_data, "val": val_data, "test": test_data}

    for split_name, dataset in splits.items():
        sub_meta = dataset["U_sub_meta"]
        face_meta = dataset["U_face_meta"]
        corner_meta = dataset["U_corners_meta"]

        # --- MASK SET 1: PARENT MASKS (Used strictly for saving grouping) ---
        is_bnd_sub = sub_meta[:, 4] == 1
        is_int_sub = ~is_bnd_sub

        parent_is_bnd_face = (face_meta[:, 2] == 0) | (face_meta[:, 2] == nselx - 1) | \
                             (face_meta[:, 3] == 0) | (face_meta[:, 3] == nsely - 1)
        parent_is_int_face = ~parent_is_bnd_face

        parent_is_bnd_corner = (corner_meta[:, 2] == 0) | (corner_meta[:, 2] == nselx - 1) | \
                               (corner_meta[:, 3] == 0) | (corner_meta[:, 3] == nsely - 1)
        parent_is_int_corner = ~parent_is_bnd_corner

        # --- MASK SET 2: GEOMETRIC MASKS (Used strictly for SVD projection) ---
        is_global_bnd_face = face_meta[:, 5] == 1
        is_global_int_face = ~is_global_bnd_face

        is_global_bnd_corner = corner_meta[:, 5] == 1
        is_global_int_corner = ~is_global_bnd_corner

        # --- A. Project Volume Fields ---
        U_sub_int_rom = project_to_rom(dataset["U_sub_mat"][is_int_sub], U_int_basis, U_int_mean)
        F_sub_int_rom = project_to_rom(dataset["F_sub_mat"][is_int_sub], F_int_basis, F_int_mean)
        
        U_sub_bnd_rom = project_to_rom(dataset["U_sub_mat"][is_bnd_sub], U_bnd_basis, U_bnd_mean)
        F_sub_bnd_rom = project_to_rom(dataset["F_sub_mat"][is_bnd_sub], F_bnd_basis, F_bnd_mean)

        # --- B. Project Faces (Into full arrays) ---
        U_face_rom_all = np.empty((face_meta.shape[0], U_face_shared_k))
        U_face_rom_all[is_global_int_face] = project_to_rom(dataset["U_face_mat"][is_global_int_face], U_face_int_basis, U_face_int_mean)
        U_face_rom_all[is_global_bnd_face] = project_to_rom(dataset["U_face_mat"][is_global_bnd_face], U_face_bnd_basis, U_face_bnd_mean)

        J_face_rom_all = np.empty((dataset["J_face_meta"].shape[0], J_face_shared_k))
        J_face_rom_all[is_global_int_face] = project_to_rom(dataset["J_face_mat"][is_global_int_face], J_face_int_basis, J_face_int_mean)
        J_face_rom_all[is_global_bnd_face] = project_to_rom(dataset["J_face_mat"][is_global_bnd_face], J_face_bnd_basis, J_face_bnd_mean)

        # --- C. Center Corners (Into full arrays) ---
        U_corners_centered_all = np.empty_like(dataset["U_corners_mat"])
        U_corners_centered_all[is_global_int_corner] = dataset["U_corners_mat"][is_global_int_corner] - U_corners_int_mean
        U_corners_centered_all[is_global_bnd_corner] = dataset["U_corners_mat"][is_global_bnd_corner] - U_corners_bnd_mean

        # --- SAVE INTERNAL DATABASE (Sliced by Parent Mask) ---
        np.savez_compressed(
            f"hdg_rom_database_internal_{split_name}_cornerflux.npz",
            U_sub_rom=U_sub_int_rom, F_sub_rom=F_sub_int_rom,
            U_face_rom=U_face_rom_all[parent_is_int_face],
            J_face_rom=J_face_rom_all[parent_is_int_face],
            U_corners_centered=U_corners_centered_all[parent_is_int_corner],
            U_sub_metadata=sub_meta[is_int_sub],
            F_sub_metadata=dataset["F_sub_meta"][is_int_sub],
            U_face_metadata=face_meta[parent_is_int_face],
            J_face_metadata=dataset["J_face_meta"][parent_is_int_face],
            U_corners_metadata=corner_meta[parent_is_int_corner],
        )

        # --- SAVE BOUNDARY DATABASE (Sliced by Parent Mask) ---
        np.savez_compressed(
            f"hdg_rom_database_boundary_{split_name}_cornerflux.npz",
            U_sub_rom=U_sub_bnd_rom, F_sub_rom=F_sub_bnd_rom,
            U_face_rom=U_face_rom_all[parent_is_bnd_face],
            J_face_rom=J_face_rom_all[parent_is_bnd_face],
            U_corners_centered=U_corners_centered_all[parent_is_bnd_corner],
            U_sub_metadata=sub_meta[is_bnd_sub],
            F_sub_metadata=dataset["F_sub_meta"][is_bnd_sub],
            U_face_metadata=face_meta[parent_is_bnd_face],
            J_face_metadata=dataset["J_face_meta"][parent_is_bnd_face],
            U_corners_metadata=corner_meta[parent_is_bnd_corner],
        )

if __name__ == "__main__":
    main()