"""
diagnose_fsub_sensitivity.py
=============================
Follow-up diagnostic for the input-block sensitivity test. The swap test
showed the trained MLP operators barely react to F_sub (~0.4-2% output
change) but react enormously to U_face/U_corners (100-500%). A parallel
U-Net that takes ONLY F_sub and predicts U_sub works well -- so F_sub is
genuinely informative. That means the MLP operators are suppressing a
useful input, not correctly ignoring a useless one.

This script runs a single check to locate WHY, without training anything:

First-layer weight-norm audit
    ---------------------------------
    Inputs are standardized (StandardScaler, unit variance) before hitting
    the network, so the L2 norm of each input column's outgoing weights in
    the first Linear layer is a direct, scale-fair measure of "how much the
    network listens to this input." We split those per-column norms into
    the F_sub block vs the U block (U_face + U_corners [+ boundary flags])
    and report the mean weight norm per column in each block. If F_sub's
    weights are near-zero relative to U's, that points at weight decay /
    optimization dynamics suppressing the F_sub pathway (an easy fix:
    lower weight_decay, warm up F_sub, or train blocks with separate LR),
    rather than a representational limit of the architecture.

Run
---
    python diagnose_fsub_sensitivity.py
    python diagnose_fsub_sensitivity.py --domains internal
    python diagnose_fsub_sensitivity.py --operators solution

Requires the same working directory layout as training_f_separate.py:
    Bases/dataset_operator_{domain}_test.csv
    trained_operators/{model}_model.pth
"""

import os
import argparse

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from training_f_separate import DD_HDG_Trainer, device


# =============================================================================
# 0. SHARED COLUMN HELPERS (mirrors training_f_separate / test_input_block_sensitivity)
# =============================================================================

def get_block_columns(df: pd.DataFrame, domain_type: str):
    """Return (f_sub_cols, u_cols) where u_cols = U_face + U_corners (+ bnd
    flags, only if domain_type == 'boundary', matching what the real
    training pipeline actually feeds the model for that domain)."""
    f_e_cols = [c for c in df.columns if 'F_sub_' in c and 'mode_' in c]
    u_f_cols = [c for c in df.columns if 'U_face_' in c and 'mode_' in c]
    u_v_cols = [c for c in df.columns if 'U_corners_' in c and 'mode_' in c]
    bnd_flag_cols = [c for c in df.columns if ('U_face_' in c or 'U_corners_' in c) and '_is_bnd' in c]

    if domain_type == 'boundary':
        u_cols = u_f_cols + u_v_cols + bnd_flag_cols
    elif domain_type == 'internal':
        u_cols = u_f_cols + u_v_cols
    else:
        raise ValueError(f"Unknown domain type: {domain_type}")

    return f_e_cols, u_cols


# =============================================================================
# 1. FIRST-LAYER WEIGHT-NORM AUDIT
# =============================================================================

def build_model_from_state_dict(state_dict):
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
    return model, linear_weight_keys[0]


def audit_first_layer_weights(domain_type: str, operator: str, flux_direction: str = None):
    model_name = f"{operator}_{domain_type}8modified" + (f"_{flux_direction}" if flux_direction else "")
    model_path = os.path.join("trained_operators", f"{model_name}_model.pth")
    if not os.path.exists(model_path):
        print(f"  ! Skipping {model_name}: {model_path} not found.")
        return None

    test_file = f"Bases/dataset_operator_{domain_type}_test.csv"
    if not os.path.exists(test_file):
        print(f"  ! Skipping {model_name}: {test_file} not found.")
        return None
    df_test = pd.read_csv(test_file)
    f_sub_cols, u_cols = get_block_columns(df_test, domain_type)
    n_f, n_u = len(f_sub_cols), len(u_cols)

    state_dict = torch.load(model_path, map_location=device)
    _, first_layer_key = build_model_from_state_dict(state_dict)
    W0 = state_dict[first_layer_key].detach().cpu().numpy()  # shape (hidden0, input_dim)

    if W0.shape[1] != n_f + n_u:
        print(f"  ! Skipping {model_name}: input_dim mismatch "
              f"(model expects {W0.shape[1]}, columns give {n_f + n_u}).")
        return None

    col_norms = np.linalg.norm(W0, axis=0)  # per-input-column L2 norm, length = input_dim
    f_norms = col_norms[:n_f]
    u_norms = col_norms[n_f:n_f + n_u]

    mean_f = f_norms.mean() if n_f > 0 else 0.0
    mean_u = u_norms.mean() if n_u > 0 else 0.0
    ratio = mean_u / (mean_f + 1e-15)

    return {
        "model": model_name,
        "n_f_sub_cols": n_f,
        "n_u_cols": n_u,
        "mean_weight_norm_f_sub": mean_f,
        "mean_weight_norm_u": mean_u,
        "u_over_f_ratio": ratio,
    }


def run_weight_analysis(domains, operators, flux_directions, out_dir="input_block_sensitivity_results"):
    print("\n" + "=" * 80)
    print("PART A: FIRST-LAYER WEIGHT-NORM AUDIT (per-input-column, mean by block)")
    print("=" * 80)
    records = []
    for domain in domains:
        if "solution" in operators:
            r = audit_first_layer_weights(domain, "solution")
            if r:
                records.append(r)
        if "flux" in operators:
            for f_dir in flux_directions:
                r = audit_first_layer_weights(domain, "flux", f_dir)
                if r:
                    records.append(r)

    if not records:
        print("  No models found to audit.")
        return None

    df = pd.DataFrame(records)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "first_layer_weight_norm_audit.csv")
    df.to_csv(csv_path, index=False)

    print(df.to_string(index=False))
    print(f"\n[INFO] Weight-norm audit saved to: {csv_path}")

    # Quick bar chart: mean weight norm per column, F_sub vs U, per model.
    os.makedirs("Plots", exist_ok=True)
    x = np.arange(len(df))
    width = 0.35
    plt.figure(figsize=(12, 5.5))
    plt.bar(x - width / 2, df["mean_weight_norm_f_sub"], width, label="F_sub (mean per column)", color="steelblue", edgecolor="black")
    plt.bar(x + width / 2, df["mean_weight_norm_u"], width, label="U_face/U_corners (mean per column)", color="coral", edgecolor="black")
    plt.xticks(x, df["model"], rotation=30, ha="right")
    plt.ylabel("Mean L2 norm of first-layer weights per input column")
    plt.title("Does the first layer even listen to F_sub? (weights on standardized inputs)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join("Plots", "first_layer_weight_norm_audit.pdf")
    plt.savefig(plot_path, format="pdf", dpi=300)
    plt.close()
    print(f"[INFO] Weight-norm bar chart saved to: {plot_path}")

    return df


# =============================================================================
# 2. MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Audit first-layer weight norms to check whether the MLP operators "
                    "are suppressing the F_sub input relative to U_face/U_corners."
    )
    parser.add_argument("--domains", nargs="+", default=["internal", "boundary"],
                         choices=["internal", "boundary"])
    parser.add_argument("--operators", nargs="+", default=["solution", "flux"],
                         choices=["solution", "flux"])
    parser.add_argument("--flux_directions", nargs="+", default=["bottom", "right", "top", "left"])
    args = parser.parse_args()

    run_weight_analysis(args.domains, args.operators, args.flux_directions)

    print("\n" + "=" * 80)
    print("DONE.")
    print("=" * 80)
    print("Interpretation guide:")
    print("  If mean_weight_norm_f_sub << mean_weight_norm_u, the first layer barely")
    print("  routes F_sub forward at all -> points at weight decay / optimization")
    print("  dynamics suppressing that pathway, not a representational limit.")


if __name__ == "__main__":
    main()
