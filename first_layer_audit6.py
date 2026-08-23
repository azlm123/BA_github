import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from dd_hdg_training6 import DD_HDG_Trainer, harmonize_df

# =============================================================================
# 1. CONFIGURATION & COLUMN PARSING
# =============================================================================
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

def get_f_dim(domain_type: str = 'internal'):
    test_file = f"Bases/dataset_operator_{domain_type}_test.csv"
    if not os.path.exists(test_file):
        test_file = f"Bases/dataset_operator_{domain_type}_train.csv"
    df = harmonize_df(pd.read_csv(test_file))
    f_cols = [c for c in df.columns if 'F_sub_' in c and 'mode_' in c]
    return len(f_cols)

# =============================================================================
# 2. AUDIT EXECUTION FOR ALL 6 OPERATOR MODELS
# =============================================================================
def audit_all_models(models_dir: str = "trained_operators"):
    domains = ["internal", "boundary"]
    flux_directions = ["bottom", "right", "top", "left"]
    
    results = []
    
    # 1. Solution Models (2 variants: internal & boundary)
    for domain in domains:
        f_dim = get_f_dim(domain)
        s_name = f"solution_{domain}6"
        
        # Check both naming conventions (with or without '6' suffix)
        s_path = os.path.join(models_dir, f"{s_name}_model.pth")
        if not os.path.exists(s_path):
            s_path = os.path.join(models_dir, f"solution_{domain}_model.pth")
        
        if os.path.exists(s_path):
            state_dict = torch.load(s_path, map_location=device)
            w1 = state_dict["network.0.weight"].detach().cpu()
            
            w_f = w1[:, :f_dim]
            w_u = w1[:, f_dim:]
            
            norm_f = torch.norm(w_f, p='fro').item()
            norm_u = torch.norm(w_u, p='fro').item()
            mean_col_f = torch.norm(w_f, p=2, dim=0).mean().item()
            mean_col_u = torch.norm(w_u, p=2, dim=0).mean().item()
            
            results.append({
                "model": s_name,
                "domain": domain,
                "operator": "solution",
                "norm_f": norm_f,
                "norm_u": norm_u,
                "mean_col_f": mean_col_f,
                "mean_col_u": mean_col_u,
                "ratio_frobenius": norm_f / (norm_u + 1e-15),
                "ratio_per_feature": mean_col_f / (mean_col_u + 1e-15)
            })
            
    # 2. Flux Models (4 variants: internal directional fluxes)
    f_dim_flux = get_f_dim('internal')
    for f_dir in flux_directions:
        j_name = f"flux_internal6_{f_dir}"
        
        # Check both naming conventions
        j_path = os.path.join(models_dir, f"{j_name}_model.pth")
        if not os.path.exists(j_path):
            j_path = os.path.join(models_dir, f"flux_internal_{f_dir}_model.pth")
        
        if os.path.exists(j_path):
            state_dict = torch.load(j_path, map_location=device)
            w1 = state_dict["network.0.weight"].detach().cpu()
            
            w_f = w1[:, :f_dim_flux]
            w_u = w1[:, f_dim_flux:]
            
            norm_f = torch.norm(w_f, p='fro').item()
            norm_u = torch.norm(w_u, p='fro').item()
            mean_col_f = torch.norm(w_f, p=2, dim=0).mean().item()
            mean_col_u = torch.norm(w_u, p=2, dim=0).mean().item()
            
            results.append({
                "model": j_name,
                "domain": "internal",
                "operator": "flux",
                "norm_f": norm_f,
                "norm_u": norm_u,
                "mean_col_f": mean_col_f,
                "mean_col_u": mean_col_u,
                "ratio_frobenius": norm_f / (norm_u + 1e-15),
                "ratio_per_feature": mean_col_f / (mean_col_u + 1e-15)
            })

    df_res = pd.DataFrame(results)
    return df_res

# =============================================================================
# 3. PLOTTING (BALKENDIAGRAMME)
# =============================================================================
def plot_first_layer_audit(df_res: pd.DataFrame, plot_dir: str = "Plots"):
    os.makedirs(plot_dir, exist_ok=True)
    
    x = np.arange(len(df_res["model"]))
    width = 0.35
    
    # ---------------------------------------------------------
    # Chart 1: Mean Column Weight Norm per Feature Group
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5.5))
    bars1 = ax.bar(x - width/2, df_res["mean_col_f"], width, label="F_sub Features ($W_F^{(1)}$)", color="steelblue", edgecolor="black")
    bars2 = ax.bar(x + width/2, df_res["mean_col_u"], width, label="Trace & Corner Features ($W_U^{(1)}$)", color="coral", edgecolor="black")
    
    ax.set_title("First Layer Weight Audit: Mean Column $L_2$-Norm per Feature (Training 6)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Mean $L_2$-Norm per Feature Column", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(df_res["model"], rotation=30, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.legend(fontsize=10)
    
    for bar in bars1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, f"{yval:.2e}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, f"{yval:.2f}", ha="center", va="bottom", fontsize=8)
        
    plt.tight_layout()
    path1 = os.path.join(plot_dir, "first_layer_weight_audit_mean_col6.pdf")
    plt.savefig(path1, format="pdf", dpi=300)
    plt.close()
    print(f"[INFO] First layer feature weight chart saved to: {path1}")
    
    # ---------------------------------------------------------
    # Chart 2: Weight Ratio (F_sub / U_traces)
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5.0))
    bars = ax.bar(df_res["model"], df_res["ratio_per_feature"], color="darkseagreen", edgecolor="black")
    
    ax.set_title("First Layer Sensitivity Ratio: $\\|W_F^{(1)}\\|_2 / \\|W_U^{(1)}\\|_2$ (Training 6)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Weight Energy Ratio ($F_{\\text{sub}}$ / Traces)", fontsize=11)
    ax.set_xticklabels(df_res["model"], rotation=30, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, f"{yval:.2e}", ha="center", va="bottom", fontsize=8.5)
        
    plt.tight_layout()
    path2 = os.path.join(plot_dir, "first_layer_weight_ratio6.pdf")
    plt.savefig(path2, format="pdf", dpi=300)
    plt.close()
    print(f"[INFO] First layer weight ratio chart saved to: {path2}")

# =============================================================================
# 4. MAIN
# =============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("RUNNING FIRST-LAYER WEIGHT AUDIT ON ALL 6 TRAINED MODELS (TRAINING 6)")
    print("=" * 80)
    
    df_audit = audit_all_models("trained_operators")
    
    if df_audit.empty:
        print("[ERROR] No trained models found in 'trained_operators/'.")
    else:
        print(df_audit[["model", "domain", "operator", "mean_col_f", "mean_col_u", "ratio_per_feature"]].to_string(index=False))
        
        os.makedirs("Bases", exist_ok=True)
        df_audit.to_csv("Bases/first_layer_weight_audit6.csv", index=False)
        print("\n[INFO] Audit table saved to 'Bases/first_layer_weight_audit6.csv'")
        
        plot_first_layer_audit(df_audit, "Plots")