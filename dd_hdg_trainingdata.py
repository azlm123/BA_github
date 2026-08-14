import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from pathlib import Path


FACE_LABELS = ["bottom", "right", "top", "left"]
CORNER_LABELS = ["bottom_left", "bottom_right", "top_right", "top_left"]


def infer_meta_names(name: str, ncols: int) -> List[str]:
    """Infer metadata column names based on column count and entity type."""
    base = ["row_index", "sample_index", "nx_index", "ny_index"]
    
    if ncols == 5:
        # Sub-element volume fields: row_idx, sample_idx, nx, ny, is_boundary
        return base + ["is_boundary"]
    elif ncols == 6:
        # Faces and Corners: row_idx, sample_idx, nx, ny, [face/corner]_index, is_global_boundary
        if "face" in name:
            extra = "face_index"
        elif "corner" in name or "corn" in name:
            extra = "corner_index"
        else:
            extra = "extra_index"
        return base + [extra, "is_global_boundary"]
    
    return base


def build_dataset_df(name: str, coeffs: np.ndarray, meta: np.ndarray) -> Tuple[pd.DataFrame, List[str]]:
    """Return a DataFrame grouped by (sample_index, nx_index, ny_index).

    Flattens faces, corners, and volume fields per sub-element into a single row record.
    """
    nmeta = meta.shape[1]
    meta_cols = infer_meta_names(name, nmeta)
    meta_df = pd.DataFrame(meta, columns=meta_cols)

    # Coefficient columns
    if coeffs.ndim == 1:
        coeffs = coeffs.reshape(-1, 1)
    n_modes = coeffs.shape[1]
    coeff_col_names = [f"mode_{i}" for i in range(n_modes)]
    coeffs_df = pd.DataFrame(coeffs, columns=coeff_col_names)

    df = pd.concat([meta_df, coeffs_df], axis=1)

    common_keys = ["sample_index", "nx_index", "ny_index"]
    extra_keys = [c for c in meta_cols if c not in common_keys and c != "row_index" and c != "is_global_boundary"]

    rows = {}
    grouped = df.groupby(common_keys, sort=False)
    
    for gkey, gdf in grouped:
        out = {}
        for ridx, row in gdf.reset_index(drop=True).iterrows():
            if extra_keys:
                extra_name = extra_keys[0]
                try:
                    extra_val = int(row[extra_name])
                except Exception:
                    extra_val = int(float(row[extra_name]))
                    
                name_lower = name.lower()
                if "face" in name_lower:
                    extra_label = FACE_LABELS[extra_val]
                elif "corner" in name_lower or "corn" in name_lower:
                    extra_label = CORNER_LABELS[extra_val]
                else:
                    extra_label = str(extra_val + 1)
                    
                top_name = f"{name}_{extra_label}"
                
                # Capture global boundary indicator per face/corner if present
                if "is_global_boundary" in row:
                    out[f"{top_name}_is_bnd"] = int(row["is_global_boundary"])
                    
                for m in range(n_modes):
                    out[f"{top_name}_mode_{m}"] = row[coeff_col_names[m]]
            else:
                # Volume fields (U_sub, F_sub)
                if "is_boundary" in row:
                    out[f"{name}_is_bnd"] = int(row["is_boundary"])
                    
                for m in range(n_modes):
                    if len(gdf) > 1:
                        sublabel = f"instance{ridx+1}_mode_{m}"
                    else:
                        sublabel = f"mode_{m}"
                    out[f"{name}_{sublabel}"] = row[coeff_col_names[m]]

        rows[gkey] = out

    records = []
    for (sample_index, nx_index, ny_index), payload in rows.items():
        record = {
            "sample_index": int(sample_index),
            "nx_index": int(nx_index),
            "ny_index": int(ny_index),
        }
        record.update(payload)
        records.append(record)

    result_df = pd.DataFrame.from_records(records)
    return result_df, coeff_col_names


def build_master_table_from_npz(npz_path: str | Path) -> pd.DataFrame:
    """Build a unified master DataFrame for a specific NPZ database split."""
    npz = np.load(npz_path, allow_pickle=True)

    # Keys created by the SVD database generation script
    datasets = [
        ("U_sub", "U_sub_rom", "U_sub_metadata"),
        ("F_sub", "F_sub_rom", "F_sub_metadata"),
        ("U_face", "U_face_rom", "U_face_metadata"),
        ("J_face", "J_face_rom", "J_face_metadata"),
        ("U_corners", "U_corners_centered", "U_corners_metadata"),
    ]

    parts = []
    for name, coeff_key, meta_key in datasets:
        if coeff_key not in npz or meta_key not in npz:
            continue
        coeffs = npz[coeff_key]
        meta = npz[meta_key]
        df_part, _ = build_dataset_df(name, coeffs, meta)
        
        for k in ["sample_index", "nx_index", "ny_index"]:
            if k not in df_part.columns:
                df_part[k] = np.nan
        parts.append(df_part)

    if not parts:
        raise RuntimeError(f"No valid datasets found in {npz_path}")

    merged = parts[0].set_index(["sample_index", "nx_index", "ny_index"])
    for p in parts[1:]:
        merged = merged.join(p.set_index(["sample_index", "nx_index", "ny_index"]), how="outer")
    merged = merged.reset_index()

    return merged


def build_operator_databases(data_dir: str | Path = "Bases"):
    """Generate training, validation, and testing tables for both Internal and Boundary learners."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)  # Ensures the Bases folder exists
    domain_types = ["internal", "boundary"]
    split_names = ["train", "val", "test"]

    for d_type in domain_types:
        for s_name in split_names:
            file_name = f"hdg_rom_database_{d_type}_{s_name}_cornerflux.npz"
            file_path = data_dir / file_name

            if not file_path.exists():
                print(f"Skipping {file_name}: file not found.")
                continue

            print(f"Processing {file_name}...")
            df = build_master_table_from_npz(file_path)

            output_csv = data_dir / f"dataset_operator_{d_type}_{s_name}_cornerflux.csv"
            df.to_csv(output_csv, index=False)
            print(f" -> Saved: {output_csv} | Shape: {df.shape}")


def main():
    # Process all NPZ databases in current directory
    build_operator_databases()


if __name__ == "__main__":
    main()