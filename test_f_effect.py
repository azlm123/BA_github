"""
test_input_block_sensitivity.py
================================
Compares two sensitivity experiments side by side:

  (A) F_sub swap    : keep U_face/U_corners (+ boundary flags) FIXED,
                       swap in another sample's F_sub.
  (B) U swap         : keep F_sub FIXED,
                       swap in another sample's U_face/U_corners (+ flags).

For each, the same base/donor pairing is used, and we measure how much the
trained operator's prediction of U_sub (solution) / J_face (flux) moves,
both in coefficient space and in reconstructed physical-field space. This
tells you which input block the network's output is actually sensitive to.

Method (identical for both blocks, only the swapped column indices differ)
----------------------------------------------------------------------
1. Load a trained operator (solution or flux) + scalers for a domain_type.
2. Pick `n_base` base samples from the test set; pick `n_donors` other
   donor samples.
3. Baseline prediction = model(base sample, untouched).
4. For each donor: copy ONLY the target block (F_sub, or U_face+U_corners
   +flags) from the donor onto the base sample, leaving the other block as
   the base sample's own values. Predict again.
5. Compare to baseline: relative change in predicted coefficients, relative
   change in reconstructed field (if POD basis available), and relative
   magnitude of the input perturbation itself.
6. Do this for BOTH blocks with the same base/donor indices, so results are
   directly comparable, and plot them side by side.

Run
---
    python test_input_block_sensitivity.py
    python test_input_block_sensitivity.py --domains internal --n_base 1 --n_donors 20
    python test_input_block_sensitivity.py --operators solution

Requires the same working directory layout as dd_hdg_training8.py:
    Bases/dataset_operator_{domain}_test.csv
    Bases/hdg_rom_bases.npz          (optional, for field-level comparison)
    trained_operators/{model}_model.pth
    trained_operators/{model}_scalers.pkl
"""

import os
import argparse
import pickle

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from dd_hdg_training8 import DD_HDG_Trainer, device, relative_error
from dd_hdg_SVD import reconstruct_from_rom


# =============================================================================
# 1. FEATURE / MODEL LOADING HELPERS
# =============================================================================

def get_feature_blocks(df: pd.DataFrame, domain_type: str = 'internal'):
    """Return column-index groups matching the order used at training time
    in extract_features_and_targets(): F_sub | U_face | U_corners | bnd flags."""
    f_e_cols = [c for c in df.columns if 'F_sub_' in c and 'mode_' in c]
    u_f_cols = [c for c in df.columns if 'U_face_' in c and 'mode_' in c]
    u_v_cols = [c for c in df.columns if 'U_corners_' in c and 'mode_' in c]
    bnd_flag_cols = [c for c in df.columns if ('U_face_' in c or 'U_corners_' in c) and '_is_bnd' in c]

    if domain_type == 'boundary':
        feature_cols = f_e_cols + u_f_cols + u_v_cols + bnd_flag_cols
    elif domain_type == 'internal':
        feature_cols = f_e_cols + u_f_cols + u_v_cols
    else:
        raise ValueError(f"Unknown domain type: {domain_type}")

    n_f, n_u, n_v, n_b = len(f_e_cols), len(u_f_cols), len(u_v_cols), len(bnd_flag_cols)

    F_idx = list(range(0, n_f))
    U_idx = list(range(n_f, n_f + n_u + n_v + n_b))  # U_face + U_corners + boundary flags

    return feature_cols, F_idx, U_idx


def get_target_cols(df: pd.DataFrame, operator: str, flux_direction: str = None):
    if operator == 'solution':
        return [c for c in df.columns if 'U_sub_' in c and 'mode_' in c]
    elif operator == 'flux':
        if flux_direction is None:
            raise ValueError("flux_direction must be specified when operator='flux'")
        return [c for c in df.columns if f'J_face_{flux_direction}_' in c and 'mode_' in c]
    raise ValueError(f"Unknown operator type: {operator}")


def build_model_from_state_dict(state_dict):
    """Reconstruct a DD_HDG_Trainer with the correct architecture directly
    from a saved state_dict (no need to hard-code hidden_dims)."""
    linear_weight_keys = sorted(
        [k for k in state_dict if k.endswith("weight")],
        key=lambda k: int(k.split(".")[1]),
    )
    dims = [tuple(state_dict[k].shape) for k in linear_weight_keys]  # (out, in)
    in_dim = dims[0][1]
    out_dim = dims[-1][0]
    hidden_dims = tuple(d[0] for d in dims[:-1])

    model = DD_HDG_Trainer(input_dim=in_dim, output_dim=out_dim, hidden_dims=hidden_dims)
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
    model = build_model_from_state_dict(state_dict)

    with open(scaler_path, "rb") as f:
        scalers = pickle.load(f)

    return model, scalers["x_scaler"], scalers["y_scaler"]


def predict(model, x_scaler, y_scaler, X_raw: np.ndarray) -> np.ndarray:
    """X_raw: (n_samples, n_features) in ORIGINAL (unscaled) units."""
    X_scaled = x_scaler.transform(X_raw)
    X_t = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        pred_norm = model(X_t).cpu().numpy()
    return y_scaler.inverse_transform(pred_norm)


def load_basis(domain_type: str, operator: str):
    npz_path = "Bases/hdg_rom_bases.npz"
    if not os.path.exists(npz_path):
        return None, None
    rom_data = np.load(npz_path)
    tag = "int" if domain_type == "internal" else "bnd"
    key_base = f"U_sub_{tag}" if operator == "solution" else f"J_face_{tag}"
    basis_key, mean_key = f"{key_base}_basis", f"{key_base}_mean"
    if basis_key not in rom_data or mean_key not in rom_data:
        return None, None
    return rom_data[basis_key], rom_data[mean_key]


# =============================================================================
# 2. CORE SENSITIVITY EXPERIMENT (single operator, single swapped block)
# =============================================================================

def run_block_sensitivity(domain_type: str, operator: str, swap_block: str,
                           flux_direction: str = None, n_base: int = 2, n_donors: int = 20,
                           seed: int = 0, base_indices=None, donor_indices=None,
                           out_dir: str = "input_block_sensitivity_results"):
    """swap_block: 'f_sub' or 'u' (U_face + U_corners + boundary flags)."""
    model_name = f"{operator}_{domain_type}8modified" + (f"_{flux_direction}" if flux_direction else "")

    test_file = f"Bases/dataset_operator_{domain_type}_test.csv"
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Missing test dataset: {test_file}")
    df_test = pd.read_csv(test_file)

    feature_cols, F_idx, U_idx = get_feature_blocks(df_test, domain_type)
    target_cols = get_target_cols(df_test, operator, flux_direction)
    swap_idx = F_idx if swap_block == "f_sub" else U_idx

    X = df_test[feature_cols].values
    n_samples = X.shape[0]

    model, x_scaler, y_scaler = load_model_and_scalers(model_name)
    basis, basis_mean = load_basis(domain_type, operator)

    # Allow caller to pass in fixed base/donor indices so BOTH block swaps
    # use the exact same sample pairing (fair comparison).
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
    """Grouped bar chart: mean output sensitivity per model, F_sub swap vs U swap."""
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
    path = os.path.join(plot_dir, "input_block_sensitivity_comparison.pdf")
    plt.savefig(path, format="pdf", dpi=300)
    plt.close()
    print(f"[INFO] Comparison bar chart saved to: {path}")

    # Same thing but normalized by the size of the perturbation actually applied
    # (relative change in output per unit relative change in input) -> a cleaner
    # "sensitivity coefficient" that isn't skewed by F_sub happening to have a
    # larger natural swap magnitude than U.
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
    path = os.path.join(plot_dir, "input_block_sensitivity_normalized.pdf")
    plt.savefig(path, format="pdf", dpi=300)
    plt.close()
    print(f"[INFO] Normalized sensitivity chart saved to: {path}")


def plot_scatter_per_model(df_all: pd.DataFrame, plot_dir: str = "Plots"):
    """Per-model scatter: output change vs input change, both blocks overlaid."""
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
        description="Compare sensitivity of U_sub/J_face predictions to F_sub vs U_face/U_corners."
    )
    parser.add_argument("--domains", nargs="+", default=["internal", "boundary"],
                         choices=["internal", "boundary"])
    parser.add_argument("--operators", nargs="+", default=["solution", "flux"],
                         choices=["solution", "flux"])
    parser.add_argument("--flux_directions", nargs="+", default=["bottom", "right", "top", "left"])
    parser.add_argument("--n_base", type=int, default=2, help="Number of base samples (fixed block).")
    parser.add_argument("--n_donors", type=int, default=20, help="Number of donor samples.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    all_frames = []

    def run_both(domain, operator, flux_direction=None):
        model_name = f"{operator}_{domain}8" + (f"_{flux_direction}" if flux_direction else "")
        print("\n" + "=" * 80)
        print(f"INPUT-BLOCK SENSITIVITY TEST: {model_name.upper()}")
        print("=" * 80)

        # First run fixes the base/donor indices; reuse them for the second
        # run so both swaps are evaluated on identical sample pairs.
        df_f, base_idx, donor_idx = run_block_sensitivity(
            domain, operator, swap_block="f_sub", flux_direction=flux_direction,
            n_base=args.n_base, n_donors=args.n_donors, seed=args.seed,
        )
        df_u, _, _ = run_block_sensitivity(
            domain, operator, swap_block="u", flux_direction=flux_direction,
            n_base=args.n_base, n_donors=args.n_donors, seed=args.seed,
            base_indices=base_idx, donor_indices=donor_idx,
        )

        mean_f = df_f["output_coeff_rel_change"].mean()
        mean_u = df_u["output_coeff_rel_change"].mean()
        print(f"  Mean output change | F_sub swap: {mean_f:.4%} | U swap: {mean_u:.4%}")

        all_frames.append(df_f)
        all_frames.append(df_u)

    for domain in args.domains:
        if "solution" in args.operators:
            run_both(domain, "solution")
        if "flux" in args.operators:
            for f_dir in args.flux_directions:
                run_both(domain, "flux", f_dir)

    df_all = pd.concat(all_frames, ignore_index=True)
    os.makedirs("input_block_sensitivity_results", exist_ok=True)
    df_all.to_csv("input_block_sensitivity_results/all_block_sensitivity.csv", index=False)

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