import os
import copy
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from dd_hdg_SVD import reconstruct_from_rom

# Device Configuration
device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
print(f"Using {device} device | PyTorch Version: {torch.__version__}")

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

NUM_EPOCHS = 300
LOSS_FN = nn.MSELoss()
USER_EARLY_STOPPER = False


# =============================================================================
# 1. HELPER CLASSES & RECONSTRUCTION UTILITIES
# =============================================================================

class EarlyStopper:
    def __init__(self, tolerance=5):
        self.tolerance = tolerance
        self.last_good_state = None
        self.prev_val_loss = float('inf')
        self.bad_epoch_count = 0

    def __call__(self, model, val_loss):
        if val_loss <= self.prev_val_loss:
            self.last_good_state = copy.deepcopy(model.state_dict())
            self.prev_val_loss = val_loss
            self.bad_epoch_count = 0
            return False
        else:
            self.bad_epoch_count += 1
            if self.bad_epoch_count >= self.tolerance:
                model.load_state_dict(self.last_good_state)
                return True
            return False


class DD_HDG_Trainer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(64, 128, 64, 32)):
        super(DD_HDG_Trainer, self).__init__()
        layers = []
        activation = nn.SiLU()
        if len(hidden_dims) > 0:
            layers.append(nn.Linear(input_dim, hidden_dims[0]))
            layers.append(activation)
            for i in range(1, len(hidden_dims)):
                layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
                layers.append(activation)
            layers.append(nn.Linear(hidden_dims[-1], output_dim))
        else:
            layers.append(nn.Linear(input_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def relative_error(y_true, y_pred):
    """Compute Frobenius relative error between true and predicted reconstructed fields."""
    norm_diff = np.linalg.norm(y_true - y_pred)
    norm_true = np.linalg.norm(y_true)
    return norm_diff / (norm_true + 1e-15)


def harmonize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize subdomain feature column names to avoid feature explosion."""
    return df.rename(columns=lambda x: x.replace('F_sub_1_', 'F_sub_').replace('U_sub_1_', 'U_sub_')
                                       .replace('F_sub_2_', 'F_sub_').replace('U_sub_2_', 'U_sub_'))


# =============================================================================
# 2. DYNAMIC DATA LOADING & COLUMN PARSING
# =============================================================================

def extract_features_and_targets(df: pd.DataFrame, operator: str, flux_direction: str = None, boundary_filter: str = None):
    """
    Extract features and targets with directional face-level filtering.
    boundary_filter:
      - 'internal_only': Keep samples where target face is internal (is_bnd == 0)
      - 'boundary_only': Keep samples where target face is boundary (is_bnd == 1)
    """
    df_filtered = df.copy()

    # Apply directional face-level filtering for flux operators
    if boundary_filter and flux_direction:
        bnd_col = f"U_face_{flux_direction}_is_bnd"
        if bnd_col in df_filtered.columns:
            if boundary_filter == "boundary_only":
                df_filtered = df_filtered[df_filtered[bnd_col] == 1].reset_index(drop=True)
            elif boundary_filter == "internal_only":
                df_filtered = df_filtered[df_filtered[bnd_col] == 0].reset_index(drop=True)

    # 1. Feature columns
    f_e_cols = [c for c in df_filtered.columns if 'F_sub_' in c and 'mode_' in c]
    u_f_cols = [c for c in df_filtered.columns if 'U_face_' in c and 'mode_' in c]
    u_v_cols = [c for c in df_filtered.columns if 'U_corners_' in c and 'mode_' in c]
    bnd_flag_cols = [c for c in df_filtered.columns if ('U_face_' in c or 'U_corners_' in c) and '_is_bnd' in c]

    feature_cols = f_e_cols + u_f_cols + u_v_cols + bnd_flag_cols
    X = df_filtered[feature_cols].values

    if operator == 'solution':
        target_cols = [c for c in df_filtered.columns if 'U_sub_' in c and 'mode_' in c]
    elif operator == 'flux':
        if flux_direction is None:
            raise ValueError("flux_direction must be specified when operator='flux'")
        target_cols = [c for c in df_filtered.columns if f'J_face_{flux_direction}_' in c and 'mode_' in c]
    else:
        raise ValueError(f"Unknown operator type: {operator}")

    y = df_filtered[target_cols].values
    return X, y, len(feature_cols), len(target_cols)


def load_dataset_split(domain_type: str, operator: str, flux_direction: str = None):
    """
    Load data splits based on domain and operator:
    - S_int: Internal dataset only
    - S_bnd: Boundary dataset only
    - F_int: Concatenate internal + boundary CSVs, filter target face for is_bnd == 0
    - F_bnd: Boundary CSV, filter target face for is_bnd == 1
    """
    def read_split(split):
        if operator == 'flux' and domain_type == 'internal':
            df_int = harmonize_df(pd.read_csv(f"Bases/dataset_operator_internal_{split}.csv"))
            df_bnd = harmonize_df(pd.read_csv(f"Bases/dataset_operator_boundary_{split}.csv"))
            return pd.concat([df_int, df_bnd], axis=0, ignore_index=True)
        else:
            return harmonize_df(pd.read_csv(f"Bases/dataset_operator_{domain_type}_{split}.csv"))

    df_train = read_split("train")
    df_val = read_split("val")
    df_test = read_split("test")

    bnd_filter = None
    if operator == 'flux':
        bnd_filter = 'boundary_only' if domain_type == 'boundary' else 'internal_only'

    X_train, y_train, input_dim, output_dim = extract_features_and_targets(df_train, operator, flux_direction, bnd_filter)
    X_val, y_val, _, _ = extract_features_and_targets(df_val, operator, flux_direction, bnd_filter)
    X_test, y_test, _, _ = extract_features_and_targets(df_test, operator, flux_direction, bnd_filter)

    # Standard Scalers (fit on training set only)
    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train)
    X_val = x_scaler.transform(X_val)
    X_test = x_scaler.transform(X_test)

    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train)
    y_val = y_scaler.transform(y_val)
    y_test = y_scaler.transform(y_test)

    # PyTorch Tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    return (X_train, y_train, X_val, y_val, X_test, y_test, 
            x_scaler, y_scaler, input_dim, output_dim)

# =============================================================================
# 3. TRAINING & VALIDATION LOOPS
# =============================================================================

def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    total_loss = 0.0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


def validation_loop(dataloader, model, loss_fn):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            total_loss += loss.item()
    return total_loss / len(dataloader)


# =============================================================================
# 4. SINGLE MODEL TRAINER FUNCTION
# =============================================================================

def train_single_operator(domain_type: str, operator: str, flux_direction: str = None, 
                          batch_size=64, lr=1e-2, hidden_dims=(64, 128, 64, 32), num_epochs=NUM_EPOCHS):
    model_name = f"{operator}_{domain_type}_train8" + (f"_{flux_direction}" if flux_direction else "")
    print("\n" + "=" * 80)
    print(f"TRAINING OPERATOR: {model_name.upper()}")
    print("=" * 80)

    # 1. Load Data
    (X_tr, y_tr, X_va, y_va, X_te, y_te, 
     x_scaler, y_scaler, in_dim, out_dim) = load_dataset_split(domain_type, operator, flux_direction)

    print(f"input_dim: {in_dim} | output_dim: {out_dim}")

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_va, y_va), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=batch_size, shuffle=False)

    # 2. Instantiate Model
    model = DD_HDG_Trainer(input_dim=in_dim, output_dim=out_dim, hidden_dims=hidden_dims).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    early_stopper = EarlyStopper(tolerance=8) if USER_EARLY_STOPPER else None

    # 3. Training Loop
    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        tr_loss = train_loop(train_loader, model, LOSS_FN, optimizer)
        va_loss = validation_loop(val_loader, model, LOSS_FN)

        train_losses.append(tr_loss)
        val_losses.append(va_loss)

        if USER_EARLY_STOPPER:
            if early_stopper(model, va_loss):
                print(f"  ✓ Early stopping triggered at epoch {epoch + 1}")
                break
        else:
            scheduler.step(va_loss)

        if (epoch + 1) % 30 == 0 or epoch == num_epochs - 1:
            curr_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1:3d}/{num_epochs} | Train Loss: {tr_loss:.6f} | Val Loss: {va_loss:.6f} | LR: {curr_lr:.2e}")

    # Plot Loss Evolution
    os.makedirs("Plots", exist_ok=True)
    plot_path = os.path.join("Plots", f"{model_name}_loss_evolution.pdf")
    
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, train_losses, label="Training Loss", linewidth=1.8)
    plt.plot(epochs_range, val_losses, label="Validation Loss", linewidth=1.8, linestyle="--")
    plt.yscale("log")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (log scale)", fontsize=12)
    plt.title(f"Loss Evolution: {model_name}", fontsize=14)
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(plot_path, format="pdf", dpi=300)
    plt.close()
    print(f"  ✓ Loss curve saved to '{plot_path}'")

    # 4. Evaluation on Test Set
    test_loss = validation_loop(test_loader, model, LOSS_FN)
    print(f"  ✓ Final Test MSE Loss (Normalized): {test_loss:.6f}")

    # 5. Inverse-transform & Reconstruction Check
    model.eval()
    with torch.no_grad():
        y_pred_norm = model(X_te).cpu().numpy()
        y_true_norm = y_te.cpu().numpy()

    y_pred = y_scaler.inverse_transform(y_pred_norm)
    y_true = y_scaler.inverse_transform(y_true_norm)

    npz_basis_path = "Bases/hdg_rom_bases.npz"
    if os.path.exists(npz_basis_path):
        rom_data = np.load(npz_basis_path)
        if operator == 'solution':
            basis = rom_data[f"U_sub_{'int' if domain_type=='internal' else 'bnd'}_basis"]
            mean = rom_data[f"U_sub_{'int' if domain_type=='internal' else 'bnd'}_mean"]
        else:
            basis = rom_data[f"J_face_{'int' if domain_type=='internal' else 'bnd'}_basis"]
            mean = rom_data[f"J_face_{'int' if domain_type=='internal' else 'bnd'}_mean"]

        y_pred_rec = reconstruct_from_rom(y_pred, basis, mean)
        y_true_rec = reconstruct_from_rom(y_true, basis, mean)
        rel_err = relative_error(y_true_rec, y_pred_rec)
        print(f"  ✓ Relative Reconstructed Field Error: {rel_err:.4e}")
    else:
        rel_err = None
        print(f"  ! Warning: Basis file {npz_basis_path} not found. Skipping physical field reconstruction.")

    # 6. Save Artifacts
    os.makedirs("trained_operators", exist_ok=True)
    model_path = os.path.join("trained_operators", f"{model_name}_model.pth")
    scaler_path = os.path.join("trained_operators", f"{model_name}_scalers.pkl")

    torch.save(model.state_dict(), model_path)
    with open(scaler_path, "wb") as f:
        pickle.dump({"x_scaler": x_scaler, "y_scaler": y_scaler}, f)

    print(f"  ✓ Model & Scalers saved to directory 'trained_operators/'")
    return test_loss, rel_err


# =============================================================================
# 5. MAIN MULTI-MODEL ORCHESTRATION
# =============================================================================

def main():
    domains = ["internal", "boundary"]
    flux_directions = ["bottom", "right", "top", "left"]
    results = []

    print("Starting Training Pipeline for 10 Operator Models...")
    
    for domain in domains:
        # 1. Solution Operator S
        test_loss, rel_err = train_single_operator(
            domain_type=domain,
            operator='solution',
            flux_direction=None,
            hidden_dims=(64, 32),
            num_epochs=300
        )
        results.append({"model": f"S_{domain}8", "test_mse": test_loss, "rel_error": rel_err})

        # 2. Four Directional Flux Operators F
        for f_dir in flux_directions:
            test_loss, rel_err = train_single_operator(
                domain_type=domain,
                operator='flux',
                flux_direction=f_dir,
                hidden_dims=(64, 128, 64, 32),
                num_epochs=120
            )
            results.append({"model": f"F_{domain}8_{f_dir}", "test_mse": test_loss, "rel_error": rel_err})

    # Summary Output Table
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY REPORT")
    print("=" * 80)
    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))

    # Create & Save Bar Chart
    plot_dir = "Plots"
    os.makedirs(plot_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: Test MSE
    bars1 = axes[0].bar(df_res["model"], df_res["test_mse"], color="steelblue", edgecolor="black")
    axes[0].set_title("Test MSE by Operator", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_xticks(range(len(df_res["model"])))
    axes[0].set_xticklabels(df_res["model"], rotation=30, ha="right")
    axes[0].grid(axis="y", linestyle="--", alpha=0.7)

    for bar in bars1:
        yval = bar.get_height()
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            yval,
            f"{yval:.2e}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Subplot 2: Relative Error
    bars2 = axes[1].bar(df_res["model"], df_res["rel_error"], color="coral", edgecolor="black")
    axes[1].set_title("Relative Error by Operator", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Relative Error")
    axes[1].set_xticks(range(len(df_res["model"])))
    axes[1].set_xticklabels(df_res["model"], rotation=30, ha="right")
    axes[1].grid(axis="y", linestyle="--", alpha=0.7)

    for bar in bars2:
        yval = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            yval,
            f"{yval:.2e}" if yval is not None and yval < 0.01 else (f"{yval:.4f}" if yval is not None else "N/A"),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    save_path = os.path.join(plot_dir, "operator_train8_summary.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n[INFO] Bar chart successfully saved to: {save_path}")


if __name__ == "__main__":
    main()