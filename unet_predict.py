# unet_predict.py
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from unet_model import SubdomainUNet
from unet_predict_statistics import reconstruct_full_solution

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_single_sample(sample_idx=10):
    csv_internal = 'Bases/dataset_operator_internal_test.csv'
    csv_boundary = 'Bases/dataset_operator_boundary_test.csv'
    
    df_int = pd.read_csv(csv_internal).rename(columns=lambda x: x.replace('F_sub_1_', 'F_sub_').replace('U_sub_1_', 'U_sub_'))
    df_bnd = pd.read_csv(csv_boundary).rename(columns=lambda x: x.replace('F_sub_2_', 'F_sub_').replace('U_sub_2_', 'U_sub_'))
    
    df_int['is_bnd'] = 0.0
    df_bnd['is_bnd'] = 1.0
    df = pd.concat([df_int, df_bnd], axis=0, ignore_index=True)
    
    sdf = df[df['sample_index'] == sample_idx].sort_values(by=['ny_index', 'nx_index']).reset_index(drop=True)
    if len(sdf) == 0:
        raise ValueError(f"Sample index {sample_idx} not found.")

    f_cols = [c for c in sdf.columns if 'F_sub_' in c and 'mode_' in c]
    u_cols = [c for c in sdf.columns if 'U_sub_' in c and 'mode_' in c]
    
    n_subx, n_suby = 8, 8
    X_grid = np.zeros((1, len(f_cols), n_suby, n_subx), dtype=np.float32)
    Y_true_grid = np.zeros((1, len(u_cols), n_suby, n_subx), dtype=np.float32)
    
    for _, row in sdf.iterrows():
        nx, ny = int(row['nx_index']), int(row['ny_index'])
        X_grid[0, :, ny, nx] = row[f_cols].values
        Y_true_grid[0, :, ny, nx] = row[u_cols].values

    is_bnd_mask = sdf['is_bnd'].values.astype(bool)

    with open("trained_operators/unet_scalers.pkl", "rb") as f:
        scalers = pickle.load(f)

    X_norm = (X_grid - scalers['x_mean']) / scalers['x_std']
    
    model = SubdomainUNet(in_ch=len(f_cols), out_ch=len(u_cols)).to(device)
    model.load_state_dict(torch.load("trained_operators/unet_benchmark_model.pth", map_location=device))
    model.eval()

    with torch.no_grad():
        x_t = torch.tensor(X_norm, dtype=torch.float32, device=device)
        pred_norm = model(x_t).cpu().numpy()

    Y_pred_grid = (pred_norm * scalers['y_std']) + scalers['y_mean']
    
    pred_rom_modes = Y_pred_grid[0].transpose(1, 2, 0).reshape(-1, len(u_cols))
    true_rom_modes = Y_true_grid[0].transpose(1, 2, 0).reshape(-1, len(u_cols))

    rom_data = np.load('Bases/hdg_rom_bases.npz')
    U_pred = reconstruct_full_solution(pred_rom_modes, is_bnd_mask, rom_data, n_subx=n_subx, n_suby=n_suby)
    U_true = reconstruct_full_solution(true_rom_modes, is_bnd_mask, rom_data, n_subx=n_subx, n_suby=n_suby)

    rel_err = np.linalg.norm(U_pred - U_true) / np.linalg.norm(U_true)
    print(f"Sample {sample_idx} Physical Field Relative Error: {rel_err:.4e}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    im0 = axes[0].imshow(U_true, origin="lower", cmap="viridis")
    axes[0].set_title(f"True (Sample {sample_idx})")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(U_pred, origin="lower", cmap="viridis")
    axes[1].set_title("U-Net Prediction")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(np.abs(U_pred - U_true), origin="lower", cmap="magma")
    axes[2].set_title(f"Absolute Error (Rel: {rel_err:.2e})")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    predict_single_sample(sample_idx=7)