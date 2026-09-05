"""
test_input_block_sensitivity6.py
================================
Compares two sensitivity experiments side by side for the 6-operator architecture:
  (A) F_sub swap : keep U_face/U_corners (+ flags) FIXED, swap F_sub.
  (B) U swap     : keep F_sub FIXED, swap U_face/U_corners (+ flags).
"""

import os
import argparse
import pickle

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from dd_hdg_training6 import DD_HDG_Trainer, get_hyperparameters, harmonize_df
from dd_hdg_SVD import reconstruct_from_rom

USE_HYPERPARAMS_CSV = True  # whether to use hyperparameters from CSV or default ones
if USE_HYPERPARAMS_CSV:
    csv_para_path = "Bases/best_hyperpara6.csv"
else:
    csv_para_path = "Bases/default_hyperpara6.csv"
# =============================================================================
# 1. FEATURE / MODEL LOADING HELPERS
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def relative_error(y_true, y_pred):
    norm_diff = np.linalg.norm(y_true - y_pred)
    norm_true = np.linalg.norm(y_true)
    return norm_diff / (norm_true + 1e-15)

def get_feature_blocks(df: pd.DataFrame):
    """Parse feature column groups matching extract_features_and_targets()."""
    f_e_cols = [c for c in df.columns if 'F_sub_' in c and 'mode_' in c]
    u_f_cols = [c for c in df.columns if 'U_face_' in c and 'mode_' in c]
    u_v_cols = [c for c in df.columns if 'U_corners_' in c and 'mode_' in c]
    bnd_flag_cols = [c for c in df.columns if ('U_face_' in c or 'U_corners_' in c) and '_is_bnd' in c]

    feature_cols = f_e_cols + u_f_cols + u_v_cols + bnd_flag_cols

    n_f = len(f_e_cols)
    n_u, n_v, n_b = len(u_f_cols), len(u_v_cols), len(bnd_flag_cols)

    F_idx = list(range(0, n_f))
    U_idx = list(range(n_f, n_f + n_u + n_v + n_b))

    return feature_cols, F_idx, U_idx


def get_target_cols(df: pd.DataFrame, operator: str, flux_direction: str = None):
    if operator == 'solution':
        return [c for c in df.columns if 'U_sub_' in c and 'mode_' in c]
    elif operator == 'flux':
        if flux_direction is None:
            raise ValueError("flux_direction must be specified when operator='flux'")
        return [c for c in df.columns if f'J_face_{flux_direction}_' in c and 'mode_' in c]
    raise ValueError(f"Unknown operator type: {operator}")


def build_model_from_state_dict(state_dict, activation_fn: str = "silu"):
    linear_weight_keys = sorted(
        [k for k in state_dict if k.endswith("weight")],
        key=lambda k: int(k.split(".")[1]),
    )
    dims = [tuple(state_dict[k].shape) for k in linear_weight_keys]
    in_dim = dims[0][1]
    out_dim = dims[-1][0]
    hidden_dims = tuple(d[0] for d in dims[:-1])

    model = DD_HDG_Trainer(input_dim=in_dim, output_dim=out_dim, hidden_dims=hidden_dims,activation_fn=activation_fn)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_model_and_scalers(model_name: str):
    model_path = os.path.join("trained_operators", f"{model_name}_model.pth")
    scaler_path = os.path.join("trained_operators", f"{model_name}_scalers.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Missing trained artifacts for '{model_name}'. "
            f"Expected '{model_path}' and '{scaler_path}'. Train the model first."
        )

    state_dict = torch.load(model_path, map_location=device)
    if model_name.startswith("solution"):
        _, _, _, activation_fn = get_hyperparameters(csv_para_path, "S_internal6" if "internal" in model_name else "S_boundary8")
    else:
        flux_dir = model_name.split("_")[-1]  # e.g., "bottom", "right", etc.
        _, _, _, activation_fn = get_hyperparameters(csv_para_path, f"F_internal6_{flux_dir}" if "internal" in model_name else f"F_boundary8_{flux_dir}")
    model = build_model_from_state_dict(state_dict, activation_fn=activation_fn)

    with open(scaler_path, "rb") as f:
        scalers = pickle.load(f)

    return model, scalers["x_scaler"], scalers["y_scaler"]


def predict(model, x_scaler, y_scaler, X_raw: np.ndarray) -> np.ndarray:
    X_scaled = x_scaler.transform(X_raw)
    X_t = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        pred_norm = model(X_t).cpu().numpy()
    return y_scaler.inverse_transform(pred_norm)


def load_basis(operator: str, domain_type: str = None):
    npz_path = "Bases/hdg_rom_bases.npz"
    if not os.path.exists(npz_path):
        return None, None
    rom_data = np.load(npz_path)

    if operator == 'solution':
        basis_key = f"U_sub_{'int' if domain_type == 'internal' else 'bnd'}_basis"
        mean_key = f"U_sub_{'int' if domain_type == 'internal' else 'bnd'}_mean"
    else:
        basis_key = "J_face_int_basis"
        mean_key = "J_face_int_mean"

    if basis_key not in rom_data or mean_key not in rom_data:
        return None, None
    return rom_data[basis_key], rom_data[mean_key]


# =============================================================================
# 2. CORE SENSITIVITY EXPERIMENT
# =============================================================================

def run_block_sensitivity(operator: str, swap_block: str, domain_type: str = None,
                          flux_direction: str = None, n_base: int = 2, n_donors: int = 20,
                          seed: int = 0, base_indices=None, donor_indices=None,
                          out_dir: str = "input_block_sensitivity_results"):
    
    if operator == 'solution':
        model_name = f"solution_{domain_type}6"+("_hyperpara" if USE_HYPERPARAMS_CSV else "")
        test_file = f"Bases/dataset_operator_{domain_type}_test.csv"
        df_test = harmonize_df(pd.read_csv(test_file))
    else:
        model_name = f"flux_internal6"+("_hyperpara" if USE_HYPERPARAMS_CSV else "")+ f"_{flux_direction}"
        # Combine datasets and filter for non-boundary faces matching training6 logic
        df_test_int = harmonize_df(pd.read_csv("Bases/dataset_operator_internal_test.csv"))
        df_test_bnd = harmonize_df(pd.read_csv("Bases/dataset_operator_boundary_test.csv"))
        df_test = pd.concat([df_test_int, df_test_bnd], ignore_index=True)

        bnd_col = f"U_face_{flux_direction}_is_bnd"
        if bnd_col in df_test.columns:
            df_test = df_test[df_test[bnd_col] == 0].reset_index(drop=True)

    feature_cols, F_idx, U_idx = get_feature_blocks(df_test)
    swap_idx = F_idx if swap_block == "f_sub" else U_idx

    X = df_test[feature_cols].values
    n_samples = X.shape[0]

    model, x_scaler, y_scaler = load_model_and_scalers(model_name)
    basis, basis_mean = load_basis(operator, domain_type)

    if base_indices is None or donor_indices is None:
        rng = np.random.default_rng(seed)
        n_base_eff = min(n_base, n_samples)
        base_indices = rng.choice(n_samples, size=n_base_eff, replace=False)
        donor_pool = np.setdiff1d(np.arange(n_samples), base_indices)
        n_donors_eff = min(n_donors, len(donor_pool))
        donor_indices = rng.choice(donor_pool, size=n_donors_eff, replace=False)

    records = []
    for b in base_indices:
        base_feat = X[b].copy()
        baseline_pred = predict(model, x_scaler, y_scaler, base_feat.reshape(1, -1))[0]

        for d in donor_indices:
            mod_feat = base_feat.copy()
            mod_feat[swap_idx] = X[d, swap_idx]

            pred = predict(model, x_scaler, y_scaler, mod_feat.reshape(1, -1))[0]

            output_rel_change = np.linalg.norm(pred - baseline_pred) / (
                np.linalg.norm(baseline_pred) + 1e-15
            )
            input_rel_change = np.linalg.norm(X[d, swap_idx] - base_feat[swap_idx]) / (
                np.linalg.norm(base_feat[swap_idx]) + 1e-15
            )

            field_rel_change = None
            if basis is not None:
                pred_rec = reconstruct_from_rom(pred.reshape(1, -1), basis, basis_mean)
                base_rec = reconstruct_from_rom(baseline_pred.reshape(1, -1), basis, basis_mean)
                field_rel_change = relative_error(base_rec, pred_rec)

            records.append({
                "model": model_name,
                "swap_block": swap_block,
                "base_idx": int(b),
                "donor_idx": int(d),
                "input_rel_change": input_rel_change,
                "output_coeff_rel_change": output_rel_change,
                "output_field_rel_change": field_rel_change,
            })

    df_res = pd.DataFrame(records)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{model_name}_{swap_block}_swap.csv")
    df_res.to_csv(csv_path, index=False)

    return df_res, base_indices, donor_indices


# =============================================================================
# 3. PLOTTING
# =============================================================================

def plot_block_comparison(df_all: pd.DataFrame, plot_dir: str = "Plots"):
    os.makedirs(plot_dir, exist_ok=True)

    pivot_coeff = df_all.pivot_table(index="model", columns="swap_block",
                                      values="output_coeff_rel_change", aggfunc="mean")
    pivot_coeff = pivot_coeff.reindex(columns=["f_sub", "u"])

    x = np.arange(len(pivot_coeff.index))
    width = 0.35

    plt.figure(figsize=(12, 5.5))
    plt.bar(x - width / 2, pivot_coeff["f_sub"], width, label="Swap F_sub (U fixed)", color="steelblue", edgecolor="black")
    plt.bar(x + width / 2, pivot_coeff["u"], width, label="Swap U_face/U_corners (F_sub fixed)", color="coral", edgecolor="black")
    plt.xticks(x, pivot_coeff.index, rotation=30, ha="right")
    plt.ylabel("Mean relative change in predicted output")
    plt.title("Which input block does each operator's output actually respond to?")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(plot_dir, "input_block_sensitivity_comparison6"+("_hyperpara" if USE_HYPERPARAMS_CSV else "")+".pdf")
    plt.savefig(path, format="pdf", dpi=300)
    plt.close()
    print(f"[INFO] Comparison bar chart saved to: {path}")

    # Normalized Sensitivity Plot
    df_all = df_all.copy()
    df_all["sensitivity_ratio"] = df_all["output_coeff_rel_change"] / (df_all["input_rel_change"] + 1e-15)
    pivot_ratio = df_all.pivot_table(index="model", columns="swap_block",
                                      values="sensitivity_ratio", aggfunc="mean")
    pivot_ratio = pivot_ratio.reindex(columns=["f_sub", "u"])

    plt.figure(figsize=(12, 5.5))
    plt.bar(x - width / 2, pivot_ratio["f_sub"], width, label="Swap F_sub (U fixed)", color="steelblue", edgecolor="black")
    plt.bar(x + width / 2, pivot_ratio["u"], width, label="Swap U_face/U_corners (F_sub fixed)", color="coral", edgecolor="black")
    plt.xticks(x, pivot_ratio.index, rotation=30, ha="right")
    plt.ylabel("Output rel. change / Input rel. change")
    plt.title("Normalized sensitivity: output change per unit of input change")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(plot_dir, "input_block_sensitivity_normalized6"+("_hyperpara" if USE_HYPERPARAMS_CSV else "")+".pdf")
    plt.savefig(path, format="pdf", dpi=300)
    plt.close()
    print(f"[INFO] Normalized sensitivity chart saved to: {path}")


def plot_scatter_per_model(df_all: pd.DataFrame, plot_dir: str = "Plots"):
    os.makedirs(plot_dir, exist_ok=True)
    for model_name, grp in df_all.groupby("model"):
        plt.figure(figsize=(6.5, 5))
        for block, color, label in [("f_sub", "steelblue", "F_sub swap"),
                                     ("u", "coral", "U_face/U_corners swap")]:
            sub = grp[grp["swap_block"] == block]
            plt.scatter(sub["input_rel_change"], sub["output_coeff_rel_change"],
                        color=color, alpha=0.7, label=label)
        plt.xlabel("Relative change in the swapped input block")
        plt.ylabel("Relative change in predicted coefficients")
        plt.title(f"Input block sensitivity — {model_name}")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        path = os.path.join(plot_dir, f"{model_name}_block_sensitivity_scatter.pdf")
        plt.savefig(path, format="pdf", dpi=300)
        plt.close()
    print(f"[INFO] Per-model scatter plots saved to '{plot_dir}/'")


# =============================================================================
# 4. MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare sensitivity of U_sub/J_face predictions for 6-operator training."
    )
    parser.add_argument("--n_base", type=int, default=2, help="Number of base samples.")
    parser.add_argument("--n_donors", type=int, default=20, help="Number of donor samples.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    all_frames = []

    def evaluate_sensitivity(operator: str, domain_type: str = None, flux_direction: str = None):
        name = f"solution_{domain_type}6"+("_hyperpara" if USE_HYPERPARAMS_CSV else "") if operator == 'solution' else f"flux_internal6"+("_hyperpara" if USE_HYPERPARAMS_CSV else "")+ f"_{flux_direction}"
        print("\n" + "=" * 80)
        print(f"INPUT-BLOCK SENSITIVITY TEST: {name.upper()}")
        print("=" * 80)

        df_f, base_idx, donor_idx = run_block_sensitivity(
            operator=operator, domain_type=domain_type, flux_direction=flux_direction,
            swap_block="f_sub", n_base=args.n_base, n_donors=args.n_donors, seed=args.seed
        )
        df_u, _, _ = run_block_sensitivity(
            operator=operator, domain_type=domain_type, flux_direction=flux_direction,
            swap_block="u", n_base=args.n_base, n_donors=args.n_donors, seed=args.seed,
            base_indices=base_idx, donor_indices=donor_idx
        )

        mean_f = df_f["output_coeff_rel_change"].mean()
        mean_u = df_u["output_coeff_rel_change"].mean()
        print(f"  Mean output change | F_sub swap: {mean_f:.4%} | U swap: {mean_u:.4%}")

        all_frames.append(df_f)
        all_frames.append(df_u)

    # 1. Solution Operators
    for domain in ["internal", "boundary"]:
        evaluate_sensitivity(operator="solution", domain_type=domain)

    # 2. Flux Operators (Internal only, combined dataset)
    for f_dir in ["bottom", "right", "top", "left"]:
        evaluate_sensitivity(operator="flux", flux_direction=f_dir)

    df_all = pd.concat(all_frames, ignore_index=True)
    os.makedirs("input_block_sensitivity_results", exist_ok=True)
    df_all.to_csv("input_block_sensitivity_results/all_block_sensitivity6"+("_hyperpara" if USE_HYPERPARAMS_CSV else "_default")+".csv", index=False)

    plot_block_comparison(df_all, "Plots")
    plot_scatter_per_model(df_all, "Plots")

    print("\n" + "=" * 80)
    print("SUMMARY: mean relative output change, F_sub swap vs U swap")
    print("=" * 80)
    print(df_all.pivot_table(index="model", columns="swap_block",
                              values="output_coeff_rel_change", aggfunc="mean")
          .reindex(columns=["f_sub", "u"]).to_string())


if __name__ == "__main__":
    main()