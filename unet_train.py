# unet_train.py
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from unet_model import SubdomainUNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_spatial_dataset(internal_csv, boundary_csv, n_subx=8, n_suby=8):
    df_int = pd.read_csv(internal_csv).rename(columns=lambda x: x.replace('F_sub_1_', 'F_sub_').replace('U_sub_1_', 'U_sub_'))
    df_bnd = pd.read_csv(boundary_csv).rename(columns=lambda x: x.replace('F_sub_2_', 'F_sub_').replace('U_sub_2_', 'U_sub_'))
    
    df_int['is_bnd'] = 0.0
    df_bnd['is_bnd'] = 1.0
    df = pd.concat([df_int, df_bnd], axis=0, ignore_index=True)
    
    f_cols = [c for c in df.columns if 'F_sub_' in c and 'mode_' in c]
    u_cols = [c for c in df.columns if 'U_sub_' in c and 'mode_' in c]
    
    sample_ids = np.sort(df['sample_index'].unique())
    n_samples = len(sample_ids)
    n_f_modes = len(f_cols)
    n_u_modes = len(u_cols)
    
    X_grid = np.zeros((n_samples, n_f_modes, n_suby, n_subx), dtype=np.float32)
    Y_grid = np.zeros((n_samples, n_u_modes, n_suby, n_subx), dtype=np.float32)
    Mask_grid = np.zeros((n_samples, 1, n_suby, n_subx), dtype=np.float32)
    
    for i, s_id in enumerate(sample_ids):
        sdf = df[df['sample_index'] == s_id]
        for _, row in sdf.iterrows():
            nx, ny = int(row['nx_index']), int(row['ny_index'])
            X_grid[i, :, ny, nx] = row[f_cols].values
            Y_grid[i, :, ny, nx] = row[u_cols].values
            Mask_grid[i, 0, ny, nx] = row['is_bnd']
            
    return X_grid, Y_grid, Mask_grid, sample_ids

def train():
    csv_train_int = 'Bases/dataset_operator_internal_train.csv'
    csv_train_bnd = 'Bases/dataset_operator_boundary_train.csv'
    
    X_train, Y_train, _, _ = load_spatial_dataset(csv_train_int, csv_train_bnd)
    
    # Per-channel standard scaling
    x_mean = X_train.mean(axis=(0, 2, 3), keepdims=True)
    x_std  = X_train.std(axis=(0, 2, 3), keepdims=True) + 1e-8
    y_mean = Y_train.mean(axis=(0, 2, 3), keepdims=True)
    y_std  = Y_train.std(axis=(0, 2, 3), keepdims=True) + 1e-8
    
    X_train_norm = (X_train - x_mean) / x_std
    Y_train_norm = (Y_train - y_mean) / y_std
    
    os.makedirs("trained_operators", exist_ok=True)
    with open("trained_operators/unet_scalers.pkl", "wb") as f:
        pickle.dump({'x_mean': x_mean, 'x_std': x_std, 'y_mean': y_mean, 'y_std': y_std}, f)
        
    dataset = TensorDataset(torch.tensor(X_train_norm), torch.tensor(Y_train_norm))
    loader  = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model = SubdomainUNet(in_ch=X_train.shape[1], out_ch=Y_train.shape[1]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
    criterion = nn.MSELoss()
    
    print(f"--- Training U-Net on {len(dataset)} samples ---")
    for epoch in range(1, 301):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
            
        scheduler.step()
        if epoch % 25 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/300 | Train MSE: {epoch_loss / len(dataset):.6e}")
            
    torch.save(model.state_dict(), "trained_operators/unet_benchmark_model.pth")
    print("Model saved to trained_operators/unet_benchmark_model.pth")

if __name__ == "__main__":
    train()