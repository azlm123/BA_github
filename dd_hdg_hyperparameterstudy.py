import os
import copy
import pickle
import itertools
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

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

LOSS_FN = nn.MSELoss()


# =============================================================================
# 1. MODEL ARCHITECTURE & METRICS
# =============================================================================

class FlexibleNeuralOperator(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(64, 128, 64, 32), activation="silu"):
        super(FlexibleNeuralOperator, self).__init__()
        layers = []
        act_fn = nn.SiLU() if activation.lower() == "silu" else (nn.GELU() if activation.lower() == "gelu" else nn.ReLU())
        
        if len(hidden_dims) > 0:
            layers.append(nn.Linear(input_dim, hidden_dims[0]))
            layers.append(act_fn)
            for i in range(1, len(hidden_dims)):
                layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
                layers.append(act_fn)
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
    """Align column names across internal and boundary data splits."""
    return df.rename(columns=lambda x: x.replace('F_sub_1_', 'F_sub_').replace('U_sub_1_', 'U_sub_')
                                       .replace('F_sub_2_', 'F_sub_').replace('U_sub_2_', 'U_sub_'))


# =============================================================================
# 2. DATA EXTRACTION & LOADING
# =============================================================================

def extract_features_and_targets(df: pd.DataFrame, operator: str, flux_direction: str = None, boundary_filter: str = None):
    """
    Extract features and targets with directional face-level boundary filtering.
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

    # 2. Target columns
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


def load_dataset(mode: str, operator: str, domain_type: str = None, flux_direction: str = None):
    """
    Unified dataset loader compatible with training6 and training8 setups:
    - mode='training6':
        * Solution: load from respective domain file.
        * Flux: concat internal + boundary, filter target face for is_bnd == 0.
    - mode='training8':
        * Solution: load from respective domain file.
        * Flux (internal): concat internal + boundary, filter target face for is_bnd == 0.
        * Flux (boundary): load boundary dataset, filter target face for is_bnd == 1.
    """
    def read_split(split):
        if (mode == 'training6' and operator == 'flux') or (mode == 'training8' and operator == 'flux' and domain_type == 'internal'):
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
        if mode == 'training6':
            bnd_filter = 'internal_only'
        elif mode == 'training8':
            bnd_filter = 'boundary_only' if domain_type == 'boundary' else 'internal_only'

    X_train, y_train, in_dim, out_dim = extract_features_and_targets(df_train, operator, flux_direction, bnd_filter)
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
            x_scaler, y_scaler, in_dim, out_dim)


# =============================================================================
# 3. TRAINING & EVALUATION ROUTINES
# =============================================================================

def train_and_eval(X_tr, y_tr, X_va, y_va, X_te, y_te, in_dim, out_dim, y_scaler, 
                   cfg, basis, mean, num_epochs=150):
    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(TensorDataset(X_va, y_va), batch_size=cfg['batch_size'], shuffle=False)
    test_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=cfg['batch_size'], shuffle=False)

    model = FlexibleNeuralOperator(
        input_dim=in_dim,
        output_dim=out_dim,
        hidden_dims=cfg['hidden_dims'],
        activation=cfg['activation']
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)

    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            loss = LOSS_FN(model(X_b), y_b)
            loss.backward()
            optimizer.step()

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                va_loss += LOSS_FN(model(X_b), y_b).item()
        va_loss /= len(val_loader)
        scheduler.step(va_loss)

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_model_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_state)
    model.eval()
    
    test_mse = 0.0
    with torch.no_grad():
        for X_b, y_b in test_loader:
            test_mse += LOSS_FN(model(X_b), y_b).item()
    test_mse /= len(test_loader)

    with torch.no_grad():
        y_pred_norm = model(X_te).cpu().numpy()
        y_true_norm = y_te.cpu().numpy()

    y_pred = y_scaler.inverse_transform(y_pred_norm)
    y_true = y_scaler.inverse_transform(y_true_norm)

    if basis is not None and mean is not None:
        y_pred_rec = reconstruct_from_rom(y_pred, basis, mean)
        y_true_rec = reconstruct_from_rom(y_true, basis, mean)
        rel_err = relative_error(y_true_rec, y_pred_rec)
    else:
        rel_err = np.nan

    return test_mse, rel_err


# =============================================================================
# 4. HYPERPARAMETER GRID SWEEP ENGINE
# =============================================================================

def run_hyperparameter_study(mode="training6", num_epochs=150):
    print("\n" + "=" * 90)
    print(f"STARTING HYPERPARAMETER STUDY FOR SETUP: {mode.upper()}")
    print("=" * 90)

    param_grid = {
        'hidden_dims': [(64, 32), (64, 128, 64, 32), (128, 128, 128)],
        'lr': [1e-2, 1e-3],
        'batch_size': [32, 64],
        'activation': ['silu', 'relu'],
        'weight_decay': [5e-4]
    }

    keys, values = zip(*param_grid.items())
    configurations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print(f"Total Hyperparameter Configurations per operator: {len(configurations)}")

    npz_path = "Bases/hdg_rom_bases.npz"
    rom_data = np.load(npz_path) if os.path.exists(npz_path) else {}

    if mode == "training6":
        tasks = [
            {"name": "S_internal6", "operator": "solution", "domain": "internal", "direction": None},
            {"name": "S_boundary6", "operator": "solution", "domain": "boundary", "direction": None},
            {"name": "F_internal6_bottom", "operator": "flux", "domain": None, "direction": "bottom"},
            {"name": "F_internal6_right", "operator": "flux", "domain": None, "direction": "right"},
        ]
    else:  # training8
        tasks = [
            {"name": "S_internal8", "operator": "solution", "domain": "internal", "direction": None},
            {"name": "S_boundary8", "operator": "solution", "domain": "boundary", "direction": None},
            {"name": "F_internal8_bottom", "operator": "flux", "domain": "internal", "direction": "bottom"},
            {"name": "F_boundary8_bottom", "operator": "flux", "domain": "boundary", "direction": "bottom"},
        ]

    all_study_results = []

    for task in tasks:
        print("\n" + "-" * 70)
        print(f"Sweeping Hyperparameters for Operator: {task['name']}")
        print("-" * 70)

        (X_tr, y_tr, X_va, y_va, X_te, y_te, 
         x_scaler, y_scaler, in_dim, out_dim) = load_dataset(
            mode=mode, operator=task['operator'], domain_type=task['domain'], flux_direction=task['direction']
        )

        basis, mean = None, None
        if len(rom_data) > 0:
            if task['operator'] == 'solution':
                basis = rom_data[f"U_sub_{'int' if task['domain']=='internal' else 'bnd'}_basis"]
                mean = rom_data[f"U_sub_{'int' if task['domain']=='internal' else 'bnd'}_mean"]
            else:
                domain_key = 'int' if (task['domain'] == 'internal' or mode == 'training6') else 'bnd'
                basis = rom_data[f"J_face_{domain_key}_basis"]
                mean = rom_data[f"J_face_{domain_key}_mean"]

        best_cfg = None
        best_rel_err = float('inf')

        for i, cfg in enumerate(configurations):
            test_mse, rel_err = train_and_eval(
                X_tr, y_tr, X_va, y_va, X_te, y_te, in_dim, out_dim, y_scaler,
                cfg, basis, mean, num_epochs=num_epochs
            )

            result_entry = {
                "operator": task['name'],
                "arch": str(cfg['hidden_dims']),
                "lr": cfg['lr'],
                "batch_size": cfg['batch_size'],
                "activation": cfg['activation'],
                "test_mse": test_mse,
                "rel_error": rel_err
            }
            all_study_results.append(result_entry)

            if rel_err < best_rel_err:
                best_rel_err = rel_err
                best_cfg = cfg

            print(f" [{i+1:2d}/{len(configurations):2d}] Arch: {str(cfg['hidden_dims']):<18} | "
                  f"LR: {cfg['lr']:.0e} | BS: {cfg['batch_size']:3d} | Act: {cfg['activation']:4s} | "
                  f"Test MSE: {test_mse:.3e} | Rel Err: {rel_err:.4e}")

        print(f"\n >>> Best Config for {task['name']}: {best_cfg} | Best Rel Err: {best_rel_err:.4e}")

    # Export CSV
    df_results = pd.DataFrame(all_study_results)
    os.makedirs("Bases", exist_ok=True)
    csv_path = f"Bases/hyperpara{mode}.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n✓ Detailed results saved to '{csv_path}'")

    # Filter and Export Best Configurations CSV
    best_per_op = df_results.loc[df_results.groupby("operator")["rel_error"].idxmin()].reset_index(drop=True)
    best_csv_path = f"Bases/best_hyperpara{mode}.csv"
    best_per_op.to_csv(best_csv_path, index=False)
    print(f"✓ Best configurations saved to '{best_csv_path}'")

    # Plot Summary Bar Chart
    best_per_op = df_results.loc[df_results.groupby("operator")["rel_error"].idxmin()]
    
    plt.figure(figsize=(10, 5))
    bars = plt.bar(best_per_op["operator"], best_per_op["rel_error"], color="teal", edgecolor="black")
    plt.yscale("log")
    plt.ylabel("Relative Reconstruction Error (Log Scale)", fontsize=11)
    plt.title(f"Best Relative Error per Operator Across Hyperparameter Search ({mode.upper()})", fontsize=12, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xticks(ticks=range(len(best_per_op["operator"])), labels=best_per_op["operator"], rotation=20, ha="right")

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval * 1.1, f"{yval:.2e}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    os.makedirs("Plots", exist_ok=True)
    plot_path = f"Plots/hyperparameter_study_{mode}_best_summary.pdf"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"✓ Summary bar chart saved to '{plot_path}'")


if __name__ == "__main__":
    # Run study for training6 setup
    run_hyperparameter_study(mode="training6", num_epochs=120)

    # Run study for training8 setup
    run_hyperparameter_study(mode="training8", num_epochs=120)