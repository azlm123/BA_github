import os
import copy
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
        activation = nn.ReLU()
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


# =============================================================================
# 2. DYNAMIC DATA LOADING & COLUMN PARSING
# =============================================================================

def extract_features_and_targets(df: pd.DataFrame, operator: str, flux_direction: str = None):
    """Dynamically parse input features (F_sub, U_face, U_corners) and targets (U_sub or J_face)."""
    # 1. Identify input feature columns
    f_e_cols = [c for c in df.columns if 'F_sub_' in c and 'mode_' in c]
    u_f_cols = [c for c in df.columns if 'U_face_' in c and 'mode_' in c]
    u_v_cols = [c for c in df.columns if 'U_corners_' in c and 'mode_' in c]
    
    # Optional: include boundary flags if present (useful for boundary operator)
    bnd_flag_cols = [c for c in df.columns if ('U_face_' in c  or 'U_corners_' in c)and '_is_bnd' in c]

    print(f"Feature Columns: {len(f_e_cols)} F_sub | {len(u_f_cols)} U_face | {len(u_v_cols)} U_corners | {len(bnd_flag_cols)} Boundary Flags")
    
    feature_cols = f_e_cols + u_f_cols + u_v_cols + bnd_flag_cols
    X = df[feature_cols].values

    # 2. Identify target columns based on operator type
    if operator == 'solution':
        target_cols = [c for c in df.columns if 'U_sub_' in c and 'mode_' in c]
    elif operator == 'flux':
        if flux_direction is None:
            raise ValueError("flux_direction must be specified when operator='flux'")
        target_cols = [c for c in df.columns if f'J_face_{flux_direction}_' in c and 'mode_' in c]
    else:
        raise ValueError(f"Unknown operator type: {operator}")

    y = df[target_cols].values
    return X, y, len(feature_cols), len(target_cols)


def load_dataset_split(domain_type: str, operator: str, flux_direction: str = None):
    """Load pre-split CSV files for train, val, and test datasets."""
    train_file = f"dataset_operator_{domain_type}_train.csv"
    val_file = f"dataset_operator_{domain_type}_val.csv"
    test_file = f"dataset_operator_{domain_type}_test.csv"

    for f in [train_file, val_file, test_file]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing required dataset file: {f}. Run database script first.")

    df_train = pd.read_csv(train_file)
    df_val = pd.read_csv(val_file)
    df_test = pd.read_csv(test_file)

    X_train, y_train, input_dim, output_dim = extract_features_and_targets(df_train, operator, flux_direction)
    X_val, y_val, _, _ = extract_features_and_targets(df_val, operator, flux_direction)
    X_test, y_test, _, _ = extract_features_and_targets(df_test, operator, flux_direction)

    # Apply Standard Scalers (fit on train set only)
    from sklearn.preprocessing import StandardScaler
    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train)
    X_val = x_scaler.transform(X_val)
    X_test = x_scaler.transform(X_test)

    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train)
    y_val = y_scaler.transform(y_val)
    y_test = y_scaler.transform(y_test)

    # Convert to PyTorch Tensors
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
                          batch_size=64, lr=1e-2, hidden_dims=(64, 128, 64, 32)):
    """Train a single NN operator model, save artifacts, and evaluate reconstruction error."""
    model_name = f"{operator}_{domain_type}8" + (f"_{flux_direction}" if flux_direction else "")
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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    early_stopper = EarlyStopper(tolerance=8) if USER_EARLY_STOPPER else None

    # 3. Training Loop
    train_losses, val_losses = [], []
    for epoch in range(NUM_EPOCHS):
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

        if (epoch + 1) % 30 == 0 or epoch == NUM_EPOCHS - 1:
            curr_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1:3d}/{NUM_EPOCHS} | Train Loss: {tr_loss:.6f} | Val Loss: {va_loss:.6f} | LR: {curr_lr:.2e}")

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

    # Load DOMAIN-SPECIFIC POD Basis for physical field reconstruction
    npz_basis_path ="hdg_rom_bases.npz"
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
        # 1. Train Solution Operator S for this domain
        test_loss, rel_err = train_single_operator(
            domain_type=domain,
            operator='solution',
            flux_direction=None,
            hidden_dims=(64, 32)
        )
        results.append({"model": f"S_{domain}", "test_mse": test_loss, "rel_error": rel_err})

        # 2. Train 4 Directional Flux Operators F for this domain
        for f_dir in flux_directions:
            test_loss, rel_err = train_single_operator(
                domain_type=domain,
                operator='flux',
                flux_direction=f_dir,
                hidden_dims=(64, 128, 64, 32)
            )
            results.append({"model": f"F_{domain}_{f_dir}", "test_mse": test_loss, "rel_error": rel_err})

    # Summary Output Table
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY REPORT")
    print("=" * 80)
    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))
    # ==========================================
    # 📊 Create & Save Bar Chart (Balkendiagramm)
    # ==========================================

    # 1. Ensure the 'Plots' directory exists
    plot_dir = "Plots"
    os.makedirs(plot_dir, exist_ok=True)

    # 2. Setup figure with 2 subplots (Test MSE & Relative Error)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: Test MSE
    bars1 = axes[0].bar(df_res["model"], df_res["test_mse"], color="steelblue", edgecolor="black")
    axes[0].set_title("Test MSE by Operator", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_xticklabels(df_res["model"], rotation=30, ha="right")
    axes[0].grid(axis="y", linestyle="--", alpha=0.7)

    # Add value labels on top of bars
    for bar in bars1:
        yval = bar.get_height()
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            yval,
            f"{yval:.2e}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Subplot 2: Relative Error
    bars2 = axes[1].bar(df_res["model"], df_res["rel_error"], color="coral", edgecolor="black")
    axes[1].set_title("Relative Error by Operator", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Relative Error")
    axes[1].set_xticklabels(df_res["model"], rotation=30, ha="right")
    axes[1].grid(axis="y", linestyle="--", alpha=0.7)

    # Add value labels on top of bars
    for bar in bars2:
        yval = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            yval,
            f"{yval:.2e}" if yval < 0.01 else f"{yval:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    # 3. Save plot into Plots/
    save_path = os.path.join(plot_dir, "operator_training8_summary.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n[INFO] Bar chart successfully saved to: {save_path}")


if __name__ == "__main__":
    main()