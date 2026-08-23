import os
import pickle
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import torch.nn as nn

from dd_hdg_face_operations import extract_neighbours_indexes
from dd_hdg_SVD import project_to_rom, reconstruct_from_rom
from dd_hdg_training6 import DD_HDG_Trainer

# =========================================================================
# DEVICE SETUP & CONFIGURATION
# =========================================================================
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
print(f"Using {device} device")

torch.cuda.manual_seed(42)

csv_internal_test_path = "Bases/dataset_operator_internal_test.csv"
csv_boundary_test_path = "Bases/dataset_operator_boundary_test.csv"

csv_internal_train_path = "Bases/dataset_operator_internal_train.csv"
csv_boundary_train_path = "Bases/dataset_operator_boundary_train.csv"
npz_rom_path = "Bases/hdg_rom_bases.npz"
models_dir = "trained_operators"
use_non_true_bnd = False
sample_idx = 10
run_optimization = True

# --- Face/corner initialization ---
# 'nearest_match'      : seed from the global closest-matching training sample's values
# 'per_subdomain_match': seed each (nx, ny) subdomain from the closest F_sub match AT THAT SAME (nx, ny)
# 'zeros'              : seed all interior faces/corners at 0
# 'constant'           : seed all interior faces/corners at interior_init_constant
# 'random'             : seed with iid Gaussian noise
interior_init_mode = "nearest_match"
interior_init_constant = 0.5
interior_init_random_mean = 0.0
interior_init_random_std = 1.0
interior_init_seed = 42

n_subx, n_suby = 8, 8


# =========================================================================
# HELPER FUNCTIONS
# =========================================================================
def load_model(
    model_name: str,
    input_dim: int,
    output_dim: int,
    hidden_dims=(64, 128, 64, 32),
) -> nn.Module:
  model_path = os.path.join(models_dir, f"{model_name}_model.pth")
  model = DD_HDG_Trainer(
      input_dim=input_dim, output_dim=output_dim, hidden_dims=hidden_dims
  )
  state_dict = torch.load(model_path, map_location=device)
  model.load_state_dict(state_dict)
  model.to(device)
  model.eval()
  return model


def load_scalers(model_name: str):
  scaler_path = os.path.join(models_dir, f"{model_name}_scalers.pkl")
  with open(scaler_path, "rb") as scaler_file:
    scalers = pickle.load(scaler_file)
  return scalers["x_scaler"], scalers["y_scaler"]


def load_svd_bases(npz_path: str):
  if not os.path.exists(npz_path):
    raise FileNotFoundError(f"Missing {npz_path}. Run dd_hdg_SVD2.py first.")
  return np.load(npz_path)


def load_combined_data_stacked(
    internal_path: str,
    boundary_path: str,
    operator: str = "solution",
    flux_direction: str = None,
    sample_index: int = 0,
):
  df_int = pd.read_csv(internal_path)
  df_bnd = pd.read_csv(boundary_path)

  df_int = df_int.rename(
      columns=lambda x: x.replace("F_sub_1_", "F_sub_").replace(
          "U_sub_1_", "U_sub_"
      )
  )
  df_bnd = df_bnd.rename(
      columns=lambda x: x.replace("F_sub_2_", "F_sub_").replace(
          "U_sub_2_", "U_sub_"
      )
  )

  if "sample_index" in df_int.columns:
    df_int = df_int[df_int["sample_index"] == sample_index]
    df_bnd = df_bnd[df_bnd["sample_index"] == sample_index]

  df_int["is_boundary_domain"] = False
  df_bnd["is_boundary_domain"] = True

  stacked_df = pd.concat([df_int, df_bnd], axis=0, ignore_index=True)

  if "ny_index" in stacked_df.columns and "nx_index" in stacked_df.columns:
    stacked_df = stacked_df.sort_values(by=["ny_index", "nx_index"]).reset_index(
        drop=True
    )

  f_e_cols = [c for c in stacked_df.columns if "F_sub_" in c and "mode_" in c]
  bnd_flag_cols = [
      c
      for c in stacked_df.columns
      if ("U_face_" in c or "U_corners_" in c) and "_is_bnd" in c
  ]

  stacked_df[bnd_flag_cols] = stacked_df[bnd_flag_cols].fillna(0.0)

  X_f = stacked_df[f_e_cols].values
  X_bnd = stacked_df[bnd_flag_cols].values
  is_bnd_mask = stacked_df["is_boundary_domain"].values

  if operator == "solution":
    u_e_cols = [c for c in stacked_df.columns if "U_sub_" in c and "mode_" in c]
    y = stacked_df[u_e_cols].values
  elif operator == "flux":
    target_cols = [
        c
        for c in stacked_df.columns
        if f"J_face_{flux_direction}_" in c and "mode_" in c
    ]
    y = stacked_df[target_cols].values

  return X_f, X_bnd, y, stacked_df, is_bnd_mask


def create_face_lookup(n_subx: int, n_suby: int):
  horizontal_id = {
      (ix, iy): uid
      for uid, (ix, iy) in enumerate(
          [(x, y) for y in range(n_suby + 1) for x in range(n_subx)]
      )
  }
  vertical_id = {
      (ix, iy): uid + len(horizontal_id)
      for uid, (ix, iy) in enumerate(
          [(x, y) for y in range(n_suby) for x in range(n_subx + 1)]
      )
  }

  rows = []
  for nx in range(n_subx):
    for ny in range(n_suby):
      rows.extend([
          (nx, ny, "bottom", horizontal_id[(nx, ny)]),
          (nx, ny, "top", horizontal_id[(nx, ny + 1)]),
          (nx, ny, "left", vertical_id[(nx, ny)]),
          (nx, ny, "right", vertical_id[(nx + 1, ny)]),
      ])
  return rows, pd.DataFrame(
      rows, columns=["nx_index", "ny_index", "direction", "n_unique"]
  )


def create_corner_lookup(n_subx: int, n_suby: int):
  corner_id = {
      (ix, iy): uid
      for uid, (ix, iy) in enumerate(
          [(x, y) for y in range(n_suby + 1) for x in range(n_subx + 1)]
      )
  }
  rows = []
  for ny in range(n_suby):
    for nx in range(n_subx):
      rows.extend([
          (nx, ny, "bottom_left", corner_id[(nx, ny)]),
          (nx, ny, "bottom_right", corner_id[(nx + 1, ny)]),
          (nx, ny, "top_left", corner_id[(nx, ny + 1)]),
          (nx, ny, "top_right", corner_id[(nx + 1, ny + 1)]),
      ])
  return rows, pd.DataFrame(
      rows, columns=["nx_index", "ny_index", "corner_position", "n_unique"]
  )


def reconstruct_full_solution(
    predicted_rom_modes,
    is_bnd_mask,
    rom_data,
    n_subx=8,
    n_suby=8,
    sub_ny=9,
    sub_nx=9,
):
  if hasattr(is_bnd_mask, "cpu"):
    is_bnd_mask = is_bnd_mask.cpu().numpy()

  is_bnd_mask = is_bnd_mask.astype(bool)
  is_int_mask = ~is_bnd_mask

  U_sub_int_basis = rom_data["U_sub_int_basis"]
  U_sub_int_mean = rom_data["U_sub_int_mean"]
  U_sub_bnd_basis = rom_data["U_sub_bnd_basis"]
  U_sub_bnd_mean = rom_data["U_sub_bnd_mean"]

  num_elements = predicted_rom_modes.shape[0]
  feature_dim = sub_ny * sub_nx
  U_sub_reconstructed = np.zeros((num_elements, feature_dim), dtype=np.float32)

  if np.any(is_int_mask):
    U_sub_reconstructed[is_int_mask] = (
        predicted_rom_modes[is_int_mask] @ U_sub_int_basis.T
    ) + U_sub_int_mean

  if np.any(is_bnd_mask):
    U_sub_reconstructed[is_bnd_mask] = (
        predicted_rom_modes[is_bnd_mask] @ U_sub_bnd_basis.T
    ) + U_sub_bnd_mean

  U_patches = U_sub_reconstructed.reshape(n_suby, n_subx, sub_ny, sub_nx)

  G_ny = n_suby * (sub_ny - 1) + 1
  G_nx = n_subx * (sub_nx - 1) + 1

  U_full = np.zeros((G_ny, G_nx), dtype=np.float32)
  weights = np.zeros((G_ny, G_nx), dtype=np.float32)

  for iy in range(n_suby):
    y_start = iy * (sub_ny - 1)
    y_end = y_start + sub_ny
    for ix in range(n_subx):
      x_start = ix * (sub_nx - 1)
      x_end = x_start + sub_nx
      U_full[y_start:y_end, x_start:x_end] += U_patches[iy, ix]
      weights[y_start:y_end, x_start:x_end] += 1.0

  U_full /= weights
  return U_full


# =========================================================================
# 1. LOAD DATA & MODELS (TRAINING6 OPERATORS)
# =========================================================================
print(f"Loading full domain for Sample Index: {sample_idx}")

x_f, x_bnd, y_S, sample_df, is_bnd_mask = load_combined_data_stacked(
    csv_internal_test_path,
    csv_boundary_test_path,
    operator="solution",
    sample_index=sample_idx,
)

_, _, y_J_left, _, _ = load_combined_data_stacked(
    csv_internal_test_path,
    csv_boundary_test_path,
    operator="flux",
    flux_direction="left",
    sample_index=sample_idx,
)
_, _, y_J_right, _, _ = load_combined_data_stacked(
    csv_internal_test_path,
    csv_boundary_test_path,
    operator="flux",
    flux_direction="right",
    sample_index=sample_idx,
)
_, _, y_J_top, _, _ = load_combined_data_stacked(
    csv_internal_test_path,
    csv_boundary_test_path,
    operator="flux",
    flux_direction="top",
    sample_index=sample_idx,
)
_, _, y_J_bottom, _, _ = load_combined_data_stacked(
    csv_internal_test_path,
    csv_boundary_test_path,
    operator="flux",
    flux_direction="bottom",
    sample_index=sample_idx,
)

bnd_mask = torch.tensor(is_bnd_mask, dtype=torch.bool, device=device)
int_mask = ~bnd_mask

X_f_const = torch.tensor(x_f, dtype=torch.float32, device=device)
X_bnd_const = torch.tensor(x_bnd, dtype=torch.float32, device=device)

latent_face_dim = 3
latent_corner_dim = 1
in_dim = (
    x_f.shape[1]
    + (4 * latent_face_dim)
    + (4 * latent_corner_dim)
    + x_bnd.shape[1]
)

# Load 2 Solution Models (Internal and Boundary)
S_int = load_model(
    "solution_internal6",
    input_dim=in_dim,
    output_dim=y_S.shape[1],
    hidden_dims=(128, 64),
)
S_bnd = load_model(
    "solution_boundary6",
    input_dim=in_dim,
    output_dim=y_S.shape[1],
    hidden_dims=(128, 64),
)

# Load 4 Directional Flux Models (Single unified model per direction from training6)
J_left = load_model(
    "flux_internal6_left",
    input_dim=in_dim,
    output_dim=y_J_left.shape[1],
    hidden_dims=(128, 64),
)
J_right = load_model(
    "flux_internal6_right",
    input_dim=in_dim,
    output_dim=y_J_right.shape[1],
    hidden_dims=(128, 64),
)
J_top = load_model(
    "flux_internal6_top",
    input_dim=in_dim,
    output_dim=y_J_top.shape[1],
    hidden_dims=(128, 64),
)
J_bottom = load_model(
    "flux_internal6_bottom",
    input_dim=in_dim,
    output_dim=y_J_bottom.shape[1],
    hidden_dims=(128, 64),
)

# Load Scalers for Solution Models
x_scaler_int, y_scaler_S_int = load_scalers("solution_internal6")
x_scaler_bnd, y_scaler_S_bnd = load_scalers("solution_boundary6")

x_mean_int = torch.tensor(x_scaler_int.mean_, dtype=torch.float32, device=device)
x_scale_int = torch.tensor(
    x_scaler_int.scale_, dtype=torch.float32, device=device
)
x_mean_bnd = torch.tensor(x_scaler_bnd.mean_, dtype=torch.float32, device=device)
x_scale_bnd = torch.tensor(
    x_scaler_bnd.scale_, dtype=torch.float32, device=device
)

# Load Input and Output Scalers for the 4 Flux Networks
scalers_dict_flux = {}
for direction in ["left", "right", "top", "bottom"]:
  x_sc, y_sc = load_scalers(f"flux_internal6_{direction}")
  scalers_dict_flux[direction] = {
      "x_scale": torch.tensor(x_sc.scale_, dtype=torch.float32, device=device),
      "x_mean": torch.tensor(x_sc.mean_, dtype=torch.float32, device=device),
      "y_scale": torch.tensor(y_sc.scale_, dtype=torch.float32, device=device),
      "y_mean": torch.tensor(y_sc.mean_, dtype=torch.float32, device=device),
  }

rom_data = load_svd_bases(npz_rom_path)
U_sub_int_basis = rom_data["U_sub_int_basis"]
U_sub_int_mean = rom_data["U_sub_int_mean"]
F_sub_int_basis = rom_data["F_sub_int_basis"]
F_sub_int_mean = rom_data["F_sub_int_mean"]
J_face_int_basis = rom_data["J_face_int_basis"]
J_face_int_mean = rom_data["J_face_int_mean"]
U_face_int_basis = rom_data["U_face_int_basis"]
U_face_int_mean = rom_data["U_face_int_mean"]
U_corners_int_mean = rom_data["U_corners_int_mean"]

U_sub_bnd_basis = rom_data["U_sub_bnd_basis"]
U_sub_bnd_mean = rom_data["U_sub_bnd_mean"]
F_sub_bnd_basis = rom_data["F_sub_bnd_basis"]
F_sub_bnd_mean = rom_data["F_sub_bnd_mean"]
J_face_bnd_basis = rom_data["J_face_bnd_basis"]
J_face_bnd_mean = rom_data["J_face_bnd_mean"]
U_face_bnd_basis = rom_data["U_face_bnd_basis"]
U_face_bnd_mean = rom_data["U_face_bnd_mean"]
U_corners_bnd_mean = rom_data["U_corners_bnd_mean"]

# =========================================================================
# 2. MESH TOPOLOGY, INDEXING & BOUNDARY EXTRACTION
# =========================================================================
face_rows, face_lookup_df = create_face_lookup(n_subx, n_suby)
corner_rows, corner_lookup_df = create_corner_lookup(n_subx, n_suby)

n_corners = (n_subx + 1) * (n_suby + 1)
n_faces = (n_subx + 1) * n_suby + n_subx * (n_suby + 1)

# Identify Boundary Faces
is_global_bottom = (face_lookup_df["ny_index"] == 0) & (
    face_lookup_df["direction"] == "bottom"
)
is_global_top = (face_lookup_df["ny_index"] == n_suby - 1) & (
    face_lookup_df["direction"] == "top"
)
is_global_left = (face_lookup_df["nx_index"] == 0) & (
    face_lookup_df["direction"] == "left"
)
is_global_right = (face_lookup_df["nx_index"] == n_subx - 1) & (
    face_lookup_df["direction"] == "right"
)

idx_bnd_bottom = torch.tensor(
    face_lookup_df[is_global_bottom]["n_unique"].values,
    dtype=torch.long,
    device=device,
)
idx_bnd_top = torch.tensor(
    face_lookup_df[is_global_top]["n_unique"].values,
    dtype=torch.long,
    device=device,
)
idx_bnd_left = torch.tensor(
    face_lookup_df[is_global_left]["n_unique"].values,
    dtype=torch.long,
    device=device,
)
idx_bnd_right = torch.tensor(
    face_lookup_df[is_global_right]["n_unique"].values,
    dtype=torch.long,
    device=device,
)

global_boundary_indices = torch.cat(
    [idx_bnd_bottom, idx_bnd_top, idx_bnd_left, idx_bnd_right]
)

all_indices = np.arange(n_faces)
is_internal = ~np.isin(all_indices, global_boundary_indices.cpu().numpy())
internal_indices = torch.tensor(
    all_indices[is_internal], dtype=torch.long, device=device
)

# Identify Boundary Corners
is_global_corner_bottom = (corner_lookup_df["ny_index"] == 0) & (
    corner_lookup_df["corner_position"].isin(["bottom_left", "bottom_right"])
)
is_global_corner_top = (corner_lookup_df["ny_index"] == n_suby - 1) & (
    corner_lookup_df["corner_position"].isin(["top_left", "top_right"])
)
is_global_corner_left = (corner_lookup_df["nx_index"] == 0) & (
    corner_lookup_df["corner_position"].isin(["bottom_left", "top_left"])
)
is_global_corner_right = (corner_lookup_df["nx_index"] == n_subx - 1) & (
    corner_lookup_df["corner_position"].isin(["bottom_right", "top_right"])
)

is_global_corner_mask = (
    is_global_corner_bottom
    | is_global_corner_top
    | is_global_corner_left
    | is_global_corner_right
)
global_corner_indices = torch.tensor(
    corner_lookup_df[is_global_corner_mask]["n_unique"].values,
    dtype=torch.long,
    device=device,
)

all_corner_indices = np.arange(n_corners)
is_internal_corner = ~np.isin(
    all_corner_indices, global_corner_indices.cpu().numpy()
)
internal_corner_indices = torch.tensor(
    all_corner_indices[is_internal_corner], dtype=torch.long, device=device
)

# Extract Real Physical Boundary Faces
bottom_df = sample_df[sample_df["ny_index"] == 0].sort_values(by="nx_index")
true_bottom = torch.tensor(
    bottom_df[[
        "U_face_bottom_mode_0",
        "U_face_bottom_mode_1",
        "U_face_bottom_mode_2",
    ]].values,
    dtype=torch.float32,
    device=device,
)
if use_non_true_bnd:
  true_bottom = torch.tensor(
      project_to_rom(
          np.zeros((bottom_df.shape[0], 5)), U_face_bnd_basis, U_face_bnd_mean
      ),
      dtype=torch.float32,
      device=device,
  )

top_df = sample_df[sample_df["ny_index"] == n_suby - 1].sort_values(
    by="nx_index"
)
true_top = torch.tensor(
    top_df[[
        "U_face_top_mode_0",
        "U_face_top_mode_1",
        "U_face_top_mode_2",
    ]].values,
    dtype=torch.float32,
    device=device,
)
if use_non_true_bnd:
  true_top = torch.tensor(
      project_to_rom(
          np.zeros((top_df.shape[0], 5)), U_face_bnd_basis, U_face_bnd_mean
      ),
      dtype=torch.float32,
      device=device,
  )

left_df = sample_df[sample_df["nx_index"] == 0].sort_values(by="ny_index")
true_left = torch.tensor(
    left_df[[
        "U_face_left_mode_0",
        "U_face_left_mode_1",
        "U_face_left_mode_2",
    ]].values,
    dtype=torch.float32,
    device=device,
)
if use_non_true_bnd:
  true_left = torch.tensor(
      project_to_rom(
          np.zeros((left_df.shape[0], 5)), U_face_bnd_basis, U_face_bnd_mean
      ),
      dtype=torch.float32,
      device=device,
  )

right_df = sample_df[sample_df["nx_index"] == n_subx - 1].sort_values(
    by="ny_index"
)
true_right = torch.tensor(
    right_df[[
        "U_face_right_mode_0",
        "U_face_right_mode_1",
        "U_face_right_mode_2",
    ]].values,
    dtype=torch.float32,
    device=device,
)
if use_non_true_bnd:
  true_right = torch.tensor(
      project_to_rom(
          np.zeros((right_df.shape[0], 5)), U_face_bnd_basis, U_face_bnd_mean
      ),
      dtype=torch.float32,
      device=device,
  )

true_global_boundaries = torch.cat(
    [true_bottom, true_top, true_left, true_right]
)

# Extract Real Physical Boundary Corners
global_corners_df = corner_lookup_df[is_global_corner_mask]
true_corners_list = []
for _, row in global_corners_df.iterrows():
  nx = row["nx_index"]
  ny = row["ny_index"]
  pos = row["corner_position"]
  subelement = sample_df[
      (sample_df["nx_index"] == nx) & (sample_df["ny_index"] == ny)
  ]
  corner_val = subelement[f"U_corners_{pos}_mode_0"].values[0]
  true_corners_list.append([corner_val])

true_global_corners = torch.tensor(
    true_corners_list, dtype=torch.float32, device=device
)
if use_non_true_bnd:
  true_global_corners = torch.tensor(
      np.zeros((len(true_corners_list), 5)) - U_corners_bnd_mean,
      dtype=torch.float32,
      device=device,
  )

# Routing Index Pairs
U_dummy = np.zeros((n_subx, n_suby, 2, 2, 1))
neighbours = extract_neighbours_indexes(U_dummy)

vertical_match_A, vertical_match_B, horizontal_match_C, horizontal_match_D = (
    [],
    [],
    [],
    [],
)
for iy in range(n_suby):
  for ix in range(n_subx):
    curr_id = iy * n_subx + ix
    if neighbours[iy, ix]["right"] != -1:
      vertical_match_A.append(curr_id)
      vertical_match_B.append(neighbours[iy, ix]["right"])
    if neighbours[iy, ix]["top"] != -1:
      horizontal_match_C.append(curr_id)
      horizontal_match_D.append(neighbours[iy, ix]["top"])

idx_batch_right_face = torch.tensor(
    vertical_match_A, dtype=torch.long, device=device
)
idx_batch_left_face = torch.tensor(
    vertical_match_B, dtype=torch.long, device=device
)
idx_batch_bottom_face = torch.tensor(
    horizontal_match_D, dtype=torch.long, device=device
)
idx_batch_top_face = torch.tensor(
    horizontal_match_C, dtype=torch.long, device=device
)

idx_dict = {
    "bottom": torch.tensor(
        face_lookup_df[face_lookup_df["direction"] == "bottom"].sort_values(
            by=["ny_index", "nx_index"]
        )["n_unique"].values,
        device=device,
    ),
    "right": torch.tensor(
        face_lookup_df[face_lookup_df["direction"] == "right"].sort_values(
            by=["ny_index", "nx_index"]
        )["n_unique"].values,
        device=device,
    ),
    "top": torch.tensor(
        face_lookup_df[face_lookup_df["direction"] == "top"].sort_values(
            by=["ny_index", "nx_index"]
        )["n_unique"].values,
        device=device,
    ),
    "left": torch.tensor(
        face_lookup_df[face_lookup_df["direction"] == "left"].sort_values(
            by=["ny_index", "nx_index"]
        )["n_unique"].values,
        device=device,
    ),
    "bl": torch.tensor(
        corner_lookup_df[
            corner_lookup_df["corner_position"] == "bottom_left"
        ].sort_values(by=["ny_index", "nx_index"])["n_unique"].values,
        device=device,
    ),
    "br": torch.tensor(
        corner_lookup_df[
            corner_lookup_df["corner_position"] == "bottom_right"
        ].sort_values(by=["ny_index", "nx_index"])["n_unique"].values,
        device=device,
    ),
    "tl": torch.tensor(
        corner_lookup_df[
            corner_lookup_df["corner_position"] == "top_left"
        ].sort_values(by=["ny_index", "nx_index"])["n_unique"].values,
        device=device,
    ),
    "tr": torch.tensor(
        corner_lookup_df[
            corner_lookup_df["corner_position"] == "top_right"
        ].sort_values(by=["ny_index", "nx_index"])["n_unique"].values,
        device=device,
    ),
}

# =========================================================================
# 3. INITIALIZE PARAMETERS (HARMONIC, DATABASE MATCHING, OR CONSTANT/RANDOM)
# =========================================================================
print("\n--- INITIALIZING VIA DATABASE / SOLVER ---")
df_train_int = pd.read_csv(csv_internal_train_path)
df_train_bnd = pd.read_csv(csv_boundary_train_path)

df_train_int = df_train_int.rename(
    columns=lambda x: x.replace("F_sub_1_", "F_sub_").replace(
        "U_sub_1_", "U_sub_"
    )
)
df_train_bnd = df_train_bnd.rename(
    columns=lambda x: x.replace("F_sub_2_", "F_sub_").replace(
        "U_sub_2_", "U_sub_"
    )
)

df_train = pd.concat([df_train_int, df_train_bnd], axis=0, ignore_index=True)
df_train = df_train.sort_values(
    by=["sample_index", "ny_index", "nx_index"]
).reset_index(drop=True)

f_cols = [c for c in df_train.columns if "F_sub_" in c and "mode_" in c]
train_samples = df_train["sample_index"].unique()

query_F_vec = X_f_const.cpu().numpy().ravel()
best_sample_id = None
min_rel_err = float("inf")

for s_id in train_samples:
  s_df = df_train[df_train["sample_index"] == s_id]
  cand_F_vec = s_df[f_cols].values.ravel()
  rel_err = np.linalg.norm(query_F_vec - cand_F_vec) / (
      np.linalg.norm(query_F_vec) + 1e-15
  )
  if rel_err < min_rel_err:
    min_rel_err = rel_err
    best_sample_id = s_id

print(
    f" -> Best matching global train sample ID: {best_sample_id} | Rel F_sub"
    f" Err: {min_rel_err:.4e}"
)

# 3a. Global Nearest Match
best_df = df_train[df_train["sample_index"] == best_sample_id].sort_values(
    by=["ny_index", "nx_index"]
).reset_index(drop=True)
init_face_values_nearest = np.zeros((n_faces, 3), dtype=np.float32)
init_corner_values_nearest = np.zeros((n_corners, 1), dtype=np.float32)

for _, row in face_lookup_df.iterrows():
  nx, ny = int(row["nx_index"]), int(row["ny_index"])
  direction, uid = row["direction"], int(row["n_unique"])
  sub_row = best_df[(best_df["nx_index"] == nx) & (best_df["ny_index"] == ny)]
  for i in range(3):
    init_face_values_nearest[uid, i] = sub_row[
        f"U_face_{direction}_mode_{i}"
    ].values[0]

for _, row in corner_lookup_df.iterrows():
  nx, ny = int(row["nx_index"]), int(row["ny_index"])
  pos, uid = row["corner_position"], int(row["n_unique"])
  sub_row = best_df[(best_df["nx_index"] == nx) & (best_df["ny_index"] == ny)]
  for i in range(1):
    init_corner_values_nearest[uid, i] = sub_row[
        f"U_corners_{pos}_mode_{i}"
    ].values[0]

# 3b. Per-Subdomain Match (Location Locked)
query_F_all = X_f_const.detach().cpu().numpy()
persub_match_lookup = {}

for i, row in sample_df.iterrows():
  nx, ny = int(row["nx_index"]), int(row["ny_index"])
  q = query_F_all[i]
  loc_pool = df_train[
      (df_train["nx_index"] == nx) & (df_train["ny_index"] == ny)
  ].reset_index(drop=True)
  loc_f = loc_pool[f_cols].values
  dists = np.linalg.norm(loc_f - q[None, :], axis=1)
  best_j = int(np.argmin(dists))
  persub_match_lookup[(nx, ny)] = loc_pool.iloc[best_j]

init_face_values_persub = np.zeros((n_faces, 3), dtype=np.float32)
init_corner_values_persub = np.zeros((n_corners, 1), dtype=np.float32)

for _, row in face_lookup_df.iterrows():
  nx, ny = int(row["nx_index"]), int(row["ny_index"])
  direction, uid = row["direction"], int(row["n_unique"])
  sub_row = persub_match_lookup[(nx, ny)]
  for i in range(3):
    init_face_values_persub[uid, i] = sub_row[f"U_face_{direction}_mode_{i}"]

for _, row in corner_lookup_df.iterrows():
  nx, ny = int(row["nx_index"]), int(row["ny_index"])
  pos, uid = row["corner_position"], int(row["n_unique"])
  sub_row = persub_match_lookup[(nx, ny)]
  for i in range(1):
    init_corner_values_persub[uid, i] = sub_row[f"U_corners_{pos}_mode_{i}"]


# 3d. Final Mode Selection
print(f"--- FACE/CORNER INIT MODE: '{interior_init_mode}' ---")
if interior_init_mode == "nearest_match":
  init_face_values = init_face_values_nearest
  init_corner_values = init_corner_values_nearest
elif interior_init_mode == "per_subdomain_match":
  init_face_values = init_face_values_persub
  init_corner_values = init_corner_values_persub
elif interior_init_mode == "zeros":
  init_face_values = np.zeros((n_faces, 3), dtype=np.float32)
  init_corner_values = np.zeros((n_corners, 1), dtype=np.float32)
elif interior_init_mode == "constant":
  init_face_values = np.full(
      (n_faces, 3), float(interior_init_constant), dtype=np.float32
  )
  init_corner_values = np.full(
      (n_corners, 1), float(interior_init_constant), dtype=np.float32
  )
elif interior_init_mode == "random":
  if interior_init_seed is not None:
    torch.manual_seed(interior_init_seed)
  init_face_values = (
      (
          interior_init_random_mean
          + interior_init_random_std * torch.randn((n_faces, 3))
      )
      .numpy()
      .astype(np.float32)
  )
  init_corner_values = (
      (
          interior_init_random_mean
          + interior_init_random_std * torch.randn((n_corners, 1))
      )
      .numpy()
      .astype(np.float32)
  )
else:
  raise ValueError(f"Unknown interior_init_mode: {interior_init_mode!r}")

# 3e. Structurally Pin Boundary DOFs
init_face_tensor = torch.tensor(
    init_face_values, dtype=torch.float32, device=device
)
init_corner_tensor = torch.tensor(
    init_corner_values, dtype=torch.float32, device=device
)

face_boundary_fixed = true_global_boundaries.detach().clone()
corner_boundary_fixed = true_global_corners.detach().clone()

interior_init_face_t = init_face_tensor[internal_indices].clone()
interior_init_corner_t = init_corner_tensor[internal_corner_indices].clone()

face_values_interior = interior_init_face_t.clone().requires_grad_()
corner_values_interior = interior_init_corner_t.clone().requires_grad_()

# =========================================================================
# 4. OPTIMIZATION CORE (UNIFIED TRAINING6 FLUX CONSERVATION)
# =========================================================================
loss_fn = nn.MSELoss(reduction="sum")


def assemble_face_corner_values():
  fv = torch.zeros((n_faces, 3), dtype=torch.float32, device=device)
  fv = fv.index_copy(0, global_boundary_indices, face_boundary_fixed)
  fv = fv.index_copy(0, internal_indices, face_values_interior)

  cv = torch.zeros((n_corners, 1), dtype=torch.float32, device=device)
  cv = cv.index_copy(0, global_corner_indices, corner_boundary_fixed)
  cv = cv.index_copy(0, internal_corner_indices, corner_values_interior)
  return fv, cv


def compute_fluxes_latent():
  face_values, corner_values = assemble_face_corner_values()

  f_bot = face_values[idx_dict["bottom"]]
  f_rig = face_values[idx_dict["right"]]
  f_top = face_values[idx_dict["top"]]
  f_lef = face_values[idx_dict["left"]]

  c_bl = corner_values[idx_dict["bl"]]
  c_br = corner_values[idx_dict["br"]]
  c_tr = corner_values[idx_dict["tr"]]
  c_tl = corner_values[idx_dict["tl"]]

  # Full concatenated input feature matrix across all 64 subdomains
  X_full = torch.cat(
      [
          X_f_const,
          f_bot,
          f_rig,
          f_top,
          f_lef,
          c_bl,
          c_br,
          c_tr,
          c_tl,
          X_bnd_const,
      ],
      dim=1,
  )

  # Scale inputs according to each unified flux operator's specific scaler
  X_scaled_left = (X_full - scalers_dict_flux["left"]["x_mean"]) / scalers_dict_flux["left"]["x_scale"]
  X_scaled_right = (X_full - scalers_dict_flux["right"]["x_mean"]) / scalers_dict_flux["right"]["x_scale"]
  X_scaled_top = (X_full - scalers_dict_flux["top"]["x_mean"]) / scalers_dict_flux["top"]["x_scale"]
  X_scaled_bottom = (X_full - scalers_dict_flux["bottom"]["x_mean"]) / scalers_dict_flux["bottom"]["x_scale"]

  # Evaluate unified models & unscale physical fluxes
  flux_left = (J_left(X_scaled_left) * scalers_dict_flux["left"]["y_scale"]) + scalers_dict_flux["left"]["y_mean"]
  flux_right = (J_right(X_scaled_right) * scalers_dict_flux["right"]["y_scale"]) + scalers_dict_flux["right"]["y_mean"]
  flux_top = (J_top(X_scaled_top) * scalers_dict_flux["top"]["y_scale"]) + scalers_dict_flux["top"]["y_mean"]
  flux_bottom = (J_bottom(X_scaled_bottom) * scalers_dict_flux["bottom"]["y_scale"]) + scalers_dict_flux["bottom"]["y_mean"]

  loss_vertical = loss_fn(
      flux_right[idx_batch_right_face], -flux_left[idx_batch_left_face]
  )
  loss_horizontal = loss_fn(
      flux_top[idx_batch_top_face], -flux_bottom[idx_batch_bottom_face]
  )

  return loss_vertical + loss_horizontal


# =========================================================================
# 5. OPTIMIZATION VIA SCIPY L-BFGS-B
# =========================================================================
def scipy_objective_latent(x_vec):
  n_face_dofs = len(internal_indices) * 3
  with torch.no_grad():
    face_values_interior.copy_(
        torch.tensor(
            x_vec[:n_face_dofs].reshape(-1, 3),
            dtype=torch.float32,
            device=device,
        )
    )
    corner_values_interior.copy_(
        torch.tensor(
            x_vec[n_face_dofs:].reshape(-1, 1),
            dtype=torch.float32,
            device=device,
        )
    )

  face_values_interior.grad = None
  corner_values_interior.grad = None

  scale = 1000.0
  loss = compute_fluxes_latent() * scale
  loss.backward()

  g_face = (
      face_values_interior.grad.detach()
      .cpu()
      .numpy()
      .ravel()
      .astype(np.float64)
  )
  g_corner = (
      corner_values_interior.grad.detach()
      .cpu()
      .numpy()
      .ravel()
      .astype(np.float64)
  )
  grad = np.concatenate([g_face, g_corner])

  return loss.item(), grad


x0 = np.concatenate([
    interior_init_face_t.detach().cpu().numpy().ravel(),
    interior_init_corner_t.detach().cpu().numpy().ravel(),
]).astype(np.float64)

if run_optimization:
  start_time = time.perf_counter()

  res = minimize(
      fun=scipy_objective_latent,
      x0=x0,
      method="L-BFGS-B",
      jac=True,
      options={"maxiter": 400, "disp": True, "gtol": 1e-8, "ftol": 1e-14},
  )

  optimization_time = time.perf_counter() - start_time
  print(
      f"BFGS Finished: Success={res.success} | Final Loss={res.fun:.6e} |"
      f" Iterations={res.nit} | Runtime: {optimization_time:.4f}s"
  )

  n_face_dofs = len(internal_indices) * 3
  with torch.no_grad():
    face_values_interior.copy_(
        torch.tensor(
            res.x[:n_face_dofs].reshape(-1, 3),
            dtype=torch.float32,
            device=device,
        )
    )
    corner_values_interior.copy_(
        torch.tensor(
            res.x[n_face_dofs:].reshape(-1, 1),
            dtype=torch.float32,
            device=device,
        )
    )
else:
  print("\n[INFO] Skipping L-BFGS-B optimization — using initial values.")
  with torch.no_grad():
    face_values_interior.copy_(interior_init_face_t)
    corner_values_interior.copy_(interior_init_corner_t)

# =========================================================================
# 6. FINAL INFERENCE & RECONSTRUCTION
# =========================================================================
print("\n--- FINAL RECONSTRUCTION ---")
with torch.no_grad():
  face_values, corner_values = assemble_face_corner_values()

  f_bot = face_values[idx_dict["bottom"]]
  f_rig = face_values[idx_dict["right"]]
  f_top = face_values[idx_dict["top"]]
  f_lef = face_values[idx_dict["left"]]

  c_bl = corner_values[idx_dict["bl"]]
  c_br = corner_values[idx_dict["br"]]
  c_tr = corner_values[idx_dict["tr"]]
  c_tl = corner_values[idx_dict["tl"]]

  # 1. Evaluate Internal Subdomains via S_int
  X_int_final = torch.cat(
      [
          X_f_const[int_mask],
          f_bot[int_mask],
          f_rig[int_mask],
          f_top[int_mask],
          f_lef[int_mask],
          c_bl[int_mask],
          c_br[int_mask],
          c_tr[int_mask],
          c_tl[int_mask],
          X_bnd_const[int_mask],
      ],
      dim=1,
  )
  X_int_final_scaled = (X_int_final - x_mean_int) / x_scale_int
  sol_int_scaled = S_int(X_int_final_scaled)

  # 2. Evaluate Boundary Subdomains via S_bnd
  X_bnd_final = torch.cat(
      [
          X_f_const[bnd_mask],
          f_bot[bnd_mask],
          f_rig[bnd_mask],
          f_top[bnd_mask],
          f_lef[bnd_mask],
          c_bl[bnd_mask],
          c_br[bnd_mask],
          c_tr[bnd_mask],
          c_tl[bnd_mask],
          X_bnd_const[bnd_mask],
      ],
      dim=1,
  )
  X_bnd_final_scaled = (X_bnd_final - x_mean_bnd) / x_scale_bnd
  sol_bnd_scaled = S_bnd(X_bnd_final_scaled)

  sol_int_modes = y_scaler_S_int.inverse_transform(sol_int_scaled.cpu().numpy())
  sol_bnd_modes = y_scaler_S_bnd.inverse_transform(sol_bnd_scaled.cpu().numpy())

  num_elements = X_f_const.shape[0]
  num_modes = sol_int_modes.shape[1]
  predicted_rom_modes = np.zeros((num_elements, num_modes))
  predicted_rom_modes[int_mask.cpu().numpy()] = sol_int_modes
  predicted_rom_modes[bnd_mask.cpu().numpy()] = sol_bnd_modes

  target_rom_modes = y_S

# =========================================================================
# 7. ROM MODES DIAGNOSTICS & HEATMAP
# =========================================================================
print("\n--- ROM MODES DIAGNOSTICS ---")
print(f"Target ROM modes shape: {target_rom_modes.shape}")
print(f"Pred ROM modes shape:   {predicted_rom_modes.shape}")

print(
    f"Min/Max of Target ROM modes: {np.min(target_rom_modes):.6f} /"
    f" {np.max(target_rom_modes):.6f}"
)
print(
    f"Min/Max of Pred ROM modes:   {np.min(predicted_rom_modes):.6f} /"
    f" {np.max(predicted_rom_modes):.6f}"
)

mode_rel_error = np.linalg.norm(predicted_rom_modes - target_rom_modes) / (
    np.linalg.norm(target_rom_modes) + 1e-15
)
print(f"\nOverall ROM Modes Relative Error: {mode_rel_error:.4e}")

if num_modes > 1:
  for m in range(num_modes):
    err_m = np.linalg.norm(
        predicted_rom_modes[:, m] - target_rom_modes[:, m]
    ) / (np.linalg.norm(target_rom_modes[:, m]) + 1e-15)
    print(f"  Mode {m} Relative Error: {err_m:.4e}")

mode_error_num = np.linalg.norm(predicted_rom_modes - target_rom_modes, axis=1)
mode_error_den = np.linalg.norm(target_rom_modes, axis=1) + 1e-15
error_bitmap_modes = np.divide(mode_error_num, mode_error_den) * 100
error_bitmap_modes = error_bitmap_modes.reshape(n_suby, n_subx)

plt.figure(figsize=(7, 6))
plt.imshow(error_bitmap_modes, origin="lower", cmap="magma")
plt.colorbar(label="Relative ROM Mode Error (%)")
plt.title("Error Bitmap (ROM Modes Comparison)")
plt.xlabel("Subdomain x index")
plt.ylabel("Subdomain y index")
plt.tight_layout()
plt.show()

# =========================================================================
# 8. PHYSICAL FIELD RECONSTRUCTION & 3/4-PANEL VISUALIZATION
# =========================================================================
U_pred_full = reconstruct_full_solution(
    predicted_rom_modes, is_bnd_mask, rom_data, n_subx=n_subx, n_suby=n_suby
)
U_true_full = reconstruct_full_solution(
    target_rom_modes, is_bnd_mask, rom_data, n_subx=n_subx, n_suby=n_suby
)

abs_error_grid = np.abs(U_pred_full - U_true_full)
rel_error_physical = np.linalg.norm(U_pred_full - U_true_full) / np.linalg.norm(
    U_true_full
)
print(
    f"Global Reconstructed Physical Solution Relative Error:"
    f" {rel_error_physical:.4e}"
)

is_nearest_match = (
    str(interior_init_mode).lower().replace(" ", "_") == "nearest_match"
)

if is_nearest_match:
  _, _, y_train_S, _, train_is_bnd_mask = load_combined_data_stacked(
      csv_internal_train_path,
      csv_boundary_train_path,
      operator="solution",
      sample_index=best_sample_id,
  )

  U_train_closest = reconstruct_full_solution(
      y_train_S, train_is_bnd_mask, rom_data, n_subx=n_subx, n_suby=n_suby
  )

  rel_train_error = np.linalg.norm(U_train_closest - U_true_full) / (
      np.linalg.norm(U_true_full) + 1e-15
  )
  print(
      f"Closest Train Sample Physical Field Relative Error:"
      f" {rel_train_error:.4e}"
  )

  abs_error_pred_vs_train = np.abs(U_pred_full - U_train_closest)
  rel_error_pred_vs_train = np.linalg.norm(
      U_pred_full - U_train_closest
  ) / (np.linalg.norm(U_train_closest) + 1e-15)
  print(
      f"Prediction vs Nearest Train Sample Relative Error:"
      f" {rel_error_pred_vs_train:.4e}"
  )

  # 2x3 Grid Plot (5 active panels)
  fig, axes = plt.subplots(2, 3, figsize=(16, 9))

  # Row 0: True Solution, Prediction, Closest Train Sample
  im0 = axes[0, 0].imshow(U_true_full, origin="lower", cmap="viridis")
  axes[0, 0].set_title(f"True Solution (Test Sample {sample_idx})")
  fig.colorbar(im0, ax=axes[0, 0])

  im1 = axes[0, 1].imshow(U_pred_full, origin="lower", cmap="viridis")
  axes[0, 1].set_title(f"Reconstructed Prediction ({interior_init_mode})")
  fig.colorbar(im1, ax=axes[0, 1])

  im2 = axes[0, 2].imshow(U_train_closest, origin="lower", cmap="viridis")
  axes[0, 2].set_title(
      f"Closest Train Sample (ID: {best_sample_id}, Rel Err:"
      f" {rel_train_error:.2e})"
  )
  fig.colorbar(im2, ax=axes[0, 2])

  # Row 1: Pred vs True Error, Pred vs Train Seed Error, Empty panel turned off
  im3 = axes[1, 0].imshow(abs_error_grid, origin="lower", cmap="magma")
  axes[1, 0].set_title(
      f"Abs Error | Pred vs True (Rel Err: {rel_error_physical:.2e})"
  )
  fig.colorbar(im3, ax=axes[1, 0])

  im4 = axes[1, 1].imshow(abs_error_pred_vs_train, origin="lower", cmap="magma")
  axes[1, 1].set_title(
      f"Abs Error | Pred vs Train Seed (Rel Err:"
      f" {rel_error_pred_vs_train:.2e})"
  )
  fig.colorbar(im4, ax=axes[1, 1])

  axes[1, 2].axis("off")

  for ax in axes.flat:
    if ax.axison:
      ax.set_xlabel("x")
      ax.set_ylabel("y")

else:
  fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

  im0 = axes[0].imshow(U_true_full, origin="lower", cmap="viridis")
  axes[0].set_title(f"True Solution (Test Sample {sample_idx})")
  fig.colorbar(im0, ax=axes[0])

  im1 = axes[1].imshow(U_pred_full, origin="lower", cmap="viridis")
  axes[1].set_title(f"Reconstructed Prediction ({interior_init_mode})")
  fig.colorbar(im1, ax=axes[1])

  im2 = axes[2].imshow(abs_error_grid, origin="lower", cmap="magma")
  axes[2].set_title(
      f"Abs Error | Pred vs True (Rel Err: {rel_error_physical:.2e})"
  )
  fig.colorbar(im2, ax=axes[2])

  for ax in axes:
    ax.set_xlabel("x")
    ax.set_ylabel("y")

os.makedirs("Plots", exist_ok=True)
plt.tight_layout()
plt.savefig(f"Plots/Inference6_2Result_{sample_idx}_{interior_init_mode}.pdf", dpi=300)
plt.show()