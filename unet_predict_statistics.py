# unet_predict_statistics.py
import os
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from unet_model import SubdomainUNet
from unet_train import load_spatial_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def reconstruct_full_solution(predicted_rom_modes, is_bnd_mask, rom_data, n_subx=8, n_suby=8, sub_ny=9, sub_nx=9):
    if hasattr(is_bnd_mask, 'cpu'):
        is_bnd_mask = is_bnd_mask.cpu().numpy()
    is_bnd_mask = is_bnd_mask.astype(bool)
    is_int_mask = ~is_bnd_mask

    U_sub_int_basis = rom_data["U_sub_int_basis"]
    U_sub_int_mean  = rom_data["U_sub_int_mean"]
    U_sub_bnd_basis = rom_data["U_sub_bnd_basis"]
    U_sub_bnd_mean  = rom_data["U_sub_bnd_mean"]

    num_elements = predicted_rom_modes.shape[0]
    feature_dim = sub_ny * sub_nx
    U_sub_reconstructed = np.zeros((num_elements, feature_dim), dtype=np.float32)

    if np.any(is_int_mask):
        U_sub_reconstructed[is_int_mask] = (predicted_rom_modes[is_int_mask] @ U_sub_int_basis.T) + U_sub_int_mean
    if np.any(is_bnd_mask):
        U_sub_reconstructed[is_bnd_mask] = (predicted_rom_modes[is_bnd_mask] @ U_sub_bnd_basis.T) + U_sub_bnd_mean

    U_patches = U_sub_reconstructed.reshape(n_suby, n_subx, sub_ny, sub_nx)
    G_ny = n_suby * (sub_ny - 1) + 1
    G_nx = n_subx * (sub_nx - 1) + 1

    U_full = np.zeros((G_ny, G_nx), dtype=np.float32)
    weights = np.zeros((G_ny, G_nx), dtype=np.float32)

    for iy in range(n_suby):
        y_start, y_end = iy * (sub_ny - 1), iy * (sub_ny - 1) + sub_ny
        for ix in range(n_subx):
            x_start, x_end = ix * (sub_nx - 1), ix * (sub_nx - 1) + sub_nx
            U_full[y_start:y_end, x_start:x_end] += U_patches[iy, ix]
            weights[y_start:y_end, x_start:x_end] += 1.0

    return U_full / weights

def evaluate():
    rom_data = np.load('Bases/hdg_rom_bases.npz')
    with open("trained_operators/unet_scalers.pkl", "rb") as f:
        scalers = pickle.load(f)
        
    x_mean, x_std = scalers['x_mean'], scalers['x_std']
    y_mean, y_std = scalers['y_mean'], scalers['y_std']
    
    csv_test_int = 'Bases/dataset_operator_internal_test.csv'
    csv_test_bnd = 'Bases/dataset_operator_boundary_test.csv'
    
    X_test, Y_test, Mask_test, sample_ids = load_spatial_dataset(csv_test_int, csv_test_bnd)
    
    model = SubdomainUNet(in_ch=X_test.shape[1], out_ch=Y_test.shape[1]).to(device)
    model.load_state_dict(torch.load("trained_operators/unet_benchmark_model.pth", map_location=device))
    model.eval()
    
    X_test_norm = (X_test - x_mean) / x_std
    with torch.no_grad():
        x_t = torch.tensor(X_test_norm, dtype=torch.float32, device=device)
        pred_norm = model(x_t).cpu().numpy()
        
    Y_pred = (pred_norm * y_std) + y_mean
    
    # Statistical Metrics
    rel_errors_modes = []
    rel_errors_physical = []
    
    for i in range(len(sample_ids)):
        # (C, ny, nx) -> (ny, nx, C) -> (n_subdomains, C)
        p_modes = Y_pred[i].transpose(1, 2, 0).reshape(-1, Y_test.shape[1])
        t_modes = Y_test[i].transpose(1, 2, 0).reshape(-1, Y_test.shape[1])
        bnd_mask = Mask_test[i, 0].ravel().astype(bool)
        
        # Mode relative error
        err_m = np.linalg.norm(p_modes - t_modes) / (np.linalg.norm(t_modes) + 1e-15)
        rel_errors_modes.append(err_m)
        
        # Full field reconstruction error
        U_pred = reconstruct_full_solution(p_modes, bnd_mask, rom_data)
        U_true = reconstruct_full_solution(t_modes, bnd_mask, rom_data)
        err_p = np.linalg.norm(U_pred - U_true) / np.linalg.norm(U_true)
        rel_errors_physical.append(err_p)
        
    print("\n--- U-NET BENCHMARK TEST SET STATISTICS ---")
    print(f"Number of Test Samples:              {len(sample_ids)}")
    print(f"Mean Relative ROM Mode Error:        {np.mean(rel_errors_modes):.4e} ± {np.std(rel_errors_modes):.4e}")
    print(f"Mean Relative Physical Field Error:  {np.mean(rel_errors_physical):.4e} ± {np.std(rel_errors_physical):.4e}")
    print(f"Median Physical Field Error:         {np.median(rel_errors_physical):.4e}")
    print(f"Min / Max Physical Field Error:      {np.min(rel_errors_physical):.4e} / {np.max(rel_errors_physical):.4e}")

    # Plot sample 0 comparison
    p_modes_0 = Y_pred[0].transpose(1, 2, 0).reshape(-1, Y_test.shape[1])
    t_modes_0 = Y_test[0].transpose(1, 2, 0).reshape(-1, Y_test.shape[1])
    bnd_mask_0 = Mask_test[0, 0].ravel().astype(bool)
    
    U_pred_0 = reconstruct_full_solution(p_modes_0, bnd_mask_0, rom_data)
    U_true_0 = reconstruct_full_solution(t_modes_0, bnd_mask_0, rom_data)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    im0 = axes[0].imshow(U_true_0, origin="lower", cmap="viridis")
    axes[0].set_title(f"True Solution (Sample {sample_ids[0]})")
    plt.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(U_pred_0, origin="lower", cmap="viridis")
    axes[1].set_title("U-Net Prediction")
    plt.colorbar(im1, ax=axes[1])
    
    im2 = axes[2].imshow(np.abs(U_pred_0 - U_true_0), origin="lower", cmap="magma")
    axes[2].set_title(f"Absolute Error (Rel: {rel_errors_physical[0]:.2e})")
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    os.makedirs("Plots", exist_ok=True)
    plt.savefig("Plots/UNetBenchmarkResult.pdf", dpi=300)
    plt.show()

if __name__ == "__main__":
    evaluate()