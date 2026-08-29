import ast
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
USE_HYPERPARAMS_CSV = True
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
    def __init__(self, input_dim, output_dim, hidden_dims=(64, 128, 64, 32),activation_fn="relu"):
        super(DD_HDG_Trainer, self).__init__()
        layers = []
        if activation_fn == "relu":
            activation = nn.ReLU()
        else:
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


def harmonize_df(df):
    """Normalize subdomain feature prefix names if multiple indices exist."""
    return df.rename(columns=lambda x: x.replace('F_sub_1_', 'F_sub_').replace('U_sub_1_', 'U_sub_')
                                       .replace('F_sub_2_', 'F_sub_').replace('U_sub_2_', 'U_sub_'))


# =============================================================================
# 2. DYNAMIC DATA LOADING & COLUMN PARSING
# =============================================================================

def extract_features_and_targets(df: pd.DataFrame, operator: str, flux_direction: str = None):
    """Dynamically parse input features (F_sub, U_face, U_corners) and targets (U_sub or J_face)."""
    # 1. Feature columns
    f_e_cols = [c for c in df.columns if 'F_sub_' in c and 'mode_' in c]
    u_f_cols = [c for c in df.columns if 'U_face_' in c and 'mode_' in c]
    u_v_cols = [c for c in df.columns if 'U_corners_' in c and 'mode_' in c]
    bnd_flag_cols = [c for c in df.columns if ('U_face_' in c or 'U_corners_' in c) and '_is_bnd' in c]

    feature_cols = f_e_cols + u_f_cols + u_v_cols + bnd_flag_cols
    X = df[feature_cols].values

    # 2. Target columns
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


def load_dataset_split(operator: str, domain_type: str = None, flux_direction: str = None):
    """
    - Solution operators: train separately on 'internal' or 'boundary' domains.
    - Flux operators: concatenate internal + boundary datasets and keep internal faces only (is_bnd == 0).
    """
    if operator == 'solution':
        if domain_type not in ['internal', 'boundary']:
            raise ValueError("domain_type must be 'internal' or 'boundary' for solution operator")

        train_file = f"Bases/dataset_operator_{domain_type}_train.csv"
        val_file = f"Bases/dataset_operator_{domain_type}_val.csv"
        test_file = f"Bases/dataset_operator_{domain_type}_test.csv"

        for f in [train_file, val_file, test_file]:
            if not os.path.exists(f):
                raise FileNotFoundError(f"Missing required dataset file: {f}.")

        df_train = harmonize_df(pd.read_csv(train_file))
        df_val = harmonize_df(pd.read_csv(val_file))
        df_test = harmonize_df(pd.read_csv(test_file))

    elif operator == 'flux':
        if flux_direction is None:
            raise ValueError("flux_direction must be provided for flux operators")

        # Combine datasets from both domains
        df_train_int = harmonize_df(pd.read_csv("Bases/dataset_operator_internal_train.csv"))
        df_train_bnd = harmonize_df(pd.read_csv("Bases/dataset_operator_boundary_train.csv"))
        df_train = pd.concat([df_train_int, df_train_bnd], ignore_index=True)

        df_val_int = harmonize_df(pd.read_csv("Bases/dataset_operator_internal_val.csv"))
        df_val_bnd = harmonize_df(pd.read_csv("Bases/dataset_operator_boundary_val.csv"))
        df_val = pd.concat([df_val_int, df_val_bnd], ignore_index=True)

        df_test_int = harmonize_df(pd.read_csv("Bases/dataset_operator_internal_test.csv"))
        df_test_bnd = harmonize_df(pd.read_csv("Bases/dataset_operator_boundary_test.csv"))
        df_test = pd.concat([df_test_int, df_test_bnd], ignore_index=True)

        # Filter strictly for internal faces
        bnd_col = f"U_face_{flux_direction}_is_bnd"
        if bnd_col in df_train.columns:
            df_train = df_train[df_train[bnd_col] == 0].reset_index(drop=True)
            df_val = df_val[df_val[bnd_col] == 0].reset_index(drop=True)
            df_test = df_test[df_test[bnd_col] == 0].reset_index(drop=True)

    else:
        raise ValueError(f"Unknown operator: {operator}")

    X_train, y_train, input_dim, output_dim = extract_features_and_targets(df_train, operator, flux_direction)
    X_val, y_val, _, _ = extract_features_and_targets(df_val, operator, flux_direction)
    X_test, y_test, _, _ = extract_features_and_targets(df_test, operator, flux_direction)

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

def train_single_operator(operator: str, domain_type: str = None, flux_direction: str = None, 
                          batch_size=64, lr=1e-2, hidden_dims=(64, 128, 64, 32),activation_fn="relu", num_epochs=NUM_EPOCHS):
    """Train a single NN operator model, save artifacts, and evaluate reconstruction error."""
    if operator == 'solution':
        model_name = f"solution_{domain_type}6"+("_hyperpara" if USE_HYPERPARAMS_CSV else "")
    else:
        model_name = f"flux_internal6"+("_hyperpara" if USE_HYPERPARAMS_CSV else "") + (f"_{flux_direction}" if flux_direction else "")

    print("\n" + "=" * 80)
    print(f"TRAINING OPERATOR: {model_name.upper()}")
    print("=" * 80)

    # 1. Load Data
    (X_tr, y_tr, X_va, y_va, X_te, y_te, 
     x_scaler, y_scaler, in_dim, out_dim) = load_dataset_split(
        operator=operator, domain_type=domain_type, flux_direction=flux_direction
    )

    print(f"input_dim: {in_dim} | output_dim: {out_dim}")

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_va, y_va), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=batch_size, shuffle=False)

    # 2. Instantiate Model
    model = DD_HDG_Trainer(input_dim=in_dim, output_dim=out_dim, hidden_dims=hidden_dims, activation_fn=activation_fn).to(device)
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

    # Load POD Basis
    npz_basis_path = "Bases/hdg_rom_bases.npz"
    if os.path.exists(npz_basis_path):
        rom_data = np.load(npz_basis_path)
        if operator == 'solution':
            basis_key = f"U_sub_{'int' if domain_type == 'internal' else 'bnd'}_basis"
            mean_key = f"U_sub_{'int' if domain_type == 'internal' else 'bnd'}_mean"
        else:
            basis_key = "J_face_int_basis"
            mean_key = "J_face_int_mean"

        basis = rom_data[basis_key]
        mean = rom_data[mean_key]

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
def get_hyperparameters(csv_path: str, model_key: str):
    """Retrieve hyperparameters from best_hyperpara6.csv or fall back to the closest operator."""
    df = pd.read_csv(csv_path)

    # 1. Exact match
    row = df[df['operator'] == model_key]
    if not row.empty:
        r = row.iloc[0]
        return ast.literal_eval(r['arch']), float(r['lr']), int(r['batch_size']), str(r['activation']).lower()

    # 2. Flux fallback
    if "flux" in model_key.lower() or model_key.startswith("F_"):
        is_vertical = any(x in model_key for x in ["top", "bottom"])
        domain = "internal" if "internal" in model_key else "boundary"

        # Directional symmetry fallback: top <-> bottom, left <-> right
        if "top" in model_key:
            alt = model_key.replace("top", "bottom")
        elif "bottom" in model_key:
            alt = model_key.replace("bottom", "top")
        elif "left" in model_key:
            alt = model_key.replace("left", "right")
        elif "right" in model_key:
            alt = model_key.replace("right", "left")
        else:
            alt = None

        if alt and not df[df['operator'] == alt].empty:
            r = df[df['operator'] == alt].iloc[0]
            print(f"  [Hyperparams] '{model_key}' -> directional fallback: '{alt}'")
            return ast.literal_eval(r['arch']), float(r['lr']), int(r['batch_size']), str(r['activation']).lower()

        # Same domain flux fallback
        same_domain = df[df['operator'].str.startswith(f"F_{domain}6")]
        if not same_domain.empty:
            r = same_domain.iloc[0]
            print(f"  [Hyperparams] '{model_key}' -> domain fallback: '{r['operator']}'")
            return ast.literal_eval(r['arch']), float(r['lr']), int(r['batch_size']), str(r['activation']).lower()

        # Cross-domain fallback
        other_domain = "boundary" if domain == "internal" else "internal"
        target_dirs = ["top", "bottom"] if is_vertical else ["left", "right"]
        for d_dir in target_dirs:
            cand = f"F_{other_domain}6_{d_dir}"
            cand_row = df[df['operator'] == cand]
            if not cand_row.empty:
                r = cand_row.iloc[0]
                print(f"  [Hyperparams] '{model_key}' -> cross-domain fallback: '{r['operator']}'")
                return ast.literal_eval(r['arch']), float(r['lr']), int(r['batch_size']), str(r['activation']).lower()

    # 3. Solution fallback: internal <-> boundary
    elif "solution" in model_key.lower() or model_key.startswith("S_"):
        alt = (
            model_key.replace("internal", "boundary")
            if "internal" in model_key
            else model_key.replace("boundary", "internal")
        )
        if not df[df['operator'] == alt].empty:
            r = df[df['operator'] == alt].iloc[0]
            print(f"  [Hyperparams] '{model_key}' -> solution fallback: '{alt}'")
            return ast.literal_eval(r['arch']), float(r['lr']), int(r['batch_size']), str(r['activation']).lower()

    # Default fallback
    r = df.iloc[0]
    return ast.literal_eval(r['arch']), float(r['lr']), int(r['batch_size']), str(r['activation']).lower()

# =============================================================================
# 5. MAIN MULTI-MODEL ORCHESTRATION
# =============================================================================

def main():
    results = []
    if USE_HYPERPARAMS_CSV:
        csv_file = "Bases/best_hyperpara6.csv"
    else:
        csv_file = "Bases/default_hyperpara6.csv"

    print("Starting Training Pipeline for 6 Operators (Loading from best_hyperpara6.csv)...")
    
    # 1. Solution Operator for Internal Subdomains
    sol_int_key = "S_internal6"
    arch, lr, bs, act = get_hyperparameters(csv_file, sol_int_key)

    print("\n" + "-" * 70)
    print(f">> Model: {sol_int_key}")
    print(f"   Architecture: {arch} | LR: {lr} | Batch Size: {bs} | Activation: {act}")
    print("-" * 70)

    test_loss, rel_err = train_single_operator(
        operator='solution',
        domain_type='internal',
        hidden_dims=arch,
        lr=lr,
        batch_size=bs,
        activation_fn=act,
        num_epochs=300
    )
    results.append({"model": sol_int_key, "test_mse": test_loss, "rel_error": rel_err})

    # 2. Solution Operator for Boundary Subdomains
    sol_bnd_key = "S_boundary6"
    arch, lr, bs, act = get_hyperparameters(csv_file, sol_bnd_key)

    print("\n" + "-" * 70)
    print(f">> Model: {sol_bnd_key}")
    print(f"   Architecture: {arch} | LR: {lr} | Batch Size: {bs} | Activation: {act}")
    print("-" * 70)

    test_loss, rel_err = train_single_operator(
        operator='solution',
        domain_type='boundary',
        hidden_dims=arch,
        lr=lr,
        batch_size=bs,
        activation_fn=act,
        num_epochs=300
    )
    results.append({"model": sol_bnd_key, "test_mse": test_loss, "rel_error": rel_err})

    # 3. Four Directional Internal Flux Operators
    flux_directions = ["bottom", "right", "top", "left"]
    for f_dir in flux_directions:
        flux_key = f"F_internal6_{f_dir}"
        arch, lr, bs, act = get_hyperparameters(csv_file, flux_key)

        print("\n" + "-" * 70)
        print(f">> Model: {flux_key}")
        print(f"   Architecture: {arch} | LR: {lr} | Batch Size: {bs} | Activation: {act}")
        print("-" * 70)

        test_loss, rel_err = train_single_operator(
            operator='flux',
            flux_direction=f_dir,
            hidden_dims=arch,
            lr=lr,
            batch_size=bs,
            activation_fn=act,
            num_epochs=100
        )
        results.append({"model": flux_key, "test_mse": test_loss, "rel_error": rel_err})

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
            fontsize=9,
        )

    # Subplot 2: Relative Error
    bars2 = axes[1].bar(df_res["model"], df_res["rel_error"], color="coral", edgecolor="black")
    axes[1].set_title("Relative Error by Operator", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Relative Error")
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
            fontsize=9,
        )

    plt.tight_layout()
    save_path = os.path.join(plot_dir, f"operator_training6"+("_hyperpara" if USE_HYPERPARAMS_CSV else "")+"_summary.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n[INFO] Bar chart successfully saved to: {save_path}")


if __name__ == "__main__":
    main()