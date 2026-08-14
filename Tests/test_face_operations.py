import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dd_hdg_face_operations import (
    subelement_boundary_local_coords,
    extract_neighbours_indexes,
    extract_boundary_values,
    extract_corner_values,
    compute_outgoing_fluxes,
    modify_subdomain_boundary_values,
    restore_subdomain_boundary_values,
)

# Small example: 4x6 element grid, 2x2 elements per patch
nely, nelx = 9, 6
nselx, nsely = 3, 3
n_samples = 1

# Patch grid dimensions
n_suby = (nely + nsely - 1) // nsely
n_subx = (nelx + nselx - 1) // nselx

# Node dimensions per patch
ny_nodes = nsely + 1
nx_nodes = nselx + 1

print(f"Global grid: nely={nely}, nelx={nelx}")
print(f"Patch size: nsely={nsely}, nselx={nselx}")
print(f"Patch grid: n_suby={n_suby}, n_subx={n_subx}")
print()

# Create dummy U_sub array
U_sub = np.random.randn(n_suby, n_subx, ny_nodes, nx_nodes, n_samples)

# Test 1: subelement_boundary_local_coords
print("=" * 60)
print("Test 1: subelement_boundary_local_coords")
print("=" * 60)
faces, corners = subelement_boundary_local_coords(nsely, nselx)
for side, coord_list in faces.items():
    print(f"{side}: {coord_list}")
for name, coord in corners.items():
    print(f"{name}: {coord}")
print()

# Test 2: extract_neighbours_indexes
print("=" * 60)
print("Test 2: extract_neighbours_indexes")
print("=" * 60)
neighbours = extract_neighbours_indexes(U_sub)
print(f"neighbours shape: {neighbours.shape}")
print(f"neighbours dtype: {neighbours.dtype}")
print()

# Print each patch and its neighbours
print("Patch neighbours detail:")
print()
for iy in range(n_suby):
    for ix in range(n_subx):
        patch_id = iy * n_subx + ix
        bottom = neighbours[iy, ix]['bottom']
        right = neighbours[iy, ix]['right']
        top = neighbours[iy, ix]['top']
        left = neighbours[iy, ix]['left']
        print(f"Patch ({iy}, {ix}) [id={patch_id}]:")
        print(f"  bottom={bottom}, right={right}, top={top}, left={left}")
print()

# Visualize patch grid as 2D array of indices
print("=" * 60)
print("PATCH GRID (n_suby x n_subx):")
print("Each cell shows the linearized patch index")
print("=" * 60)
patch_grid = np.zeros((n_suby, n_subx), dtype=int)
for iy in range(n_suby):
    for ix in range(n_subx):
        patch_grid[iy, ix] = iy * n_subx + ix
print(np.flipud(patch_grid))
print()
print("PATCH GRID VISUALIZATION (top row first):")
print(np.flipud(patch_grid))
print()

# Visualize global element grid
print("=" * 60)
print("GLOBAL ELEMENT GRID (nely x nelx):")
print("Each cell shows the linearized element index (ex*nely + ey)")
print("=" * 60)
element_grid = np.zeros((nely, nelx), dtype=int)
for ey in range(nely):
    for ex in range(nelx):
        element_grid[ey, ex] = ex * nely + ey
print("Raw grid (bottom row first):")
print(element_grid)
print("Visual grid (top row first):")
print(np.flipud(element_grid))
print()

# Map which patch each global element belongs to
print("=" * 60)
print("PATCH ASSIGNMENT (nely x nelx):")
print("Shows which patch index each element belongs to")
print("=" * 60)
patch_assignment = np.zeros((nely, nelx), dtype=int)
for iy in range(n_suby):
    for ix in range(n_subx):
        for j in range(nsely):
            for i in range(nselx):
                ey = iy * nsely + j
                ex = ix * nselx + i
                if ey < nely and ex < nelx:
                    patch_id = iy * n_subx + ix
                    patch_assignment[ey, ex] = patch_id
print("Raw assignment (bottom row first):")
print(patch_assignment)
print("Visual assignment (top row first):")
print(np.flipud(patch_assignment))
print()

# Verification: print local-to-global element mapping for a few patches
print("=" * 60)
print("VERIFICATION: Local element indices per patch")
print("=" * 60)
for iy in range(n_suby):
    for ix in range(n_subx):
        patch_id = iy * n_subx + ix
        print(f"\nPatch ({iy}, {ix}) [id={patch_id}] local elements map to global as:")
        for j in range(nsely):
            for i in range(nselx):
                ey = iy * nsely + j
                ex = ix * nselx + i
                if ey < nely and ex < nelx:
                    global_el = ex * nely + ey
                    print(f"  Local ({j}, {i}) -> Global ({ey}, {ex}) [id={global_el}]")

# Test 3: extract face nodes from U_sub using subelement_boundary_local_coords
print()
print("=" * 60)
print("Test 3: face-node extraction with subelement_boundary_local_coords")
print("=" * 60)

# Build deterministic test field once and reuse in subsequent tests.
dx = 1.0
rn = nsely + 1
cn = nselx + 1
U_test = np.zeros((n_suby, n_subx, rn, cn, 1), dtype=float)
for iy in range(n_suby):
    for ix in range(n_subx):
        for r in range(rn):
            for c in range(cn):
                U_test[iy, ix, r, c, 0] = 1000.0 * iy + 100.0 * ix + 10.0 * r + c

faces, corners = subelement_boundary_local_coords(nsely, nselx)

# Select one patch/sample and extract face values using (local_row, local_col) coordinates.
iy_face, ix_face, sample_idx = 1, 1, 0

face_nodes_by_coords = {
    side: np.array([U_test[iy_face, ix_face, r, c, sample_idx] for (r, c) in rc_list], dtype=float)
    for side, rc_list in faces.items()
}

corner_nodes_by_coords = {
    name: U_test[iy_face, ix_face, r, c, sample_idx]
    for name, (r, c) in corners.items()
}

# Direct extraction for comparison (U_sub shape:
# [patch_row, patch_col, local_row, local_col, sample_n]).
expected_face_nodes = {
    "bottom": U_test[iy_face, ix_face, 0, 1:cn - 1, sample_idx],
    "top": U_test[iy_face, ix_face, rn - 1, 1:cn - 1, sample_idx],
    "left": U_test[iy_face, ix_face, 1:rn - 1, 0, sample_idx],
    "right": U_test[iy_face, ix_face, 1:rn - 1, cn - 1, sample_idx],
}

for side in ["bottom", "right", "top", "left"]:
    print(f"{side} coords: {faces[side]}")
    print(f"Extracted {side} nodes:", face_nodes_by_coords[side])
    print(f"Expected  {side} nodes:", expected_face_nodes[side])

expected_corner_nodes = {
    "bottom_left": U_test[iy_face, ix_face, 0, 0, sample_idx],
    "bottom_right": U_test[iy_face, ix_face, 0, cn - 1, sample_idx],
    "top_left": U_test[iy_face, ix_face, rn - 1, 0, sample_idx],
    "top_right": U_test[iy_face, ix_face, rn - 1, cn - 1, sample_idx],
}

for name in ["bottom_left", "bottom_right", "top_left", "top_right"]:
    print(f"{name} coord: {corners[name]}")
    print(f"Extracted {name} node:", corner_nodes_by_coords[name])
    print(f"Expected  {name} node:", expected_corner_nodes[name])

ok_face_nodes = all(
    np.array_equal(face_nodes_by_coords[side], expected_face_nodes[side])
    for side in ["bottom", "right", "top", "left"]
)
ok_corner_nodes = all(
    np.isclose(corner_nodes_by_coords[name], expected_corner_nodes[name])
    for name in ["bottom_left", "bottom_right", "top_left", "top_right"]
)
print()
print("Face-node extraction test passed:", ok_face_nodes)
print("Corner extraction test passed:", ok_corner_nodes)
if not (ok_face_nodes and ok_corner_nodes):
    raise AssertionError("subelement_boundary_local_coords face/corner extraction test failed")

# Test 3b: extract_boundary_values and extract_corner_values
print()
print("=" * 60)
print("Test 3b: extract_boundary_values and extract_corner_values")
print("=" * 60)

boundary_values = extract_boundary_values(U_test)
corner_values = extract_corner_values(U_test)

ok_boundary_values = all(
    np.array_equal(boundary_values[side][iy_face, ix_face, :, 0], expected_face_nodes[side])
    for side in ["bottom", "right", "top", "left"]
)
ok_corner_values = all(
    np.isclose(corner_values[name][iy_face, ix_face, 0], expected_corner_nodes[name])
    for name in ["bottom_left", "bottom_right", "top_left", "top_right"]
)

print("Boundary extractor test passed:", ok_boundary_values)
print("Corner-value extractor test passed:", ok_corner_values)
if not (ok_boundary_values and ok_corner_values):
    raise AssertionError("extract_boundary_values or extract_corner_values test failed")

# Test 4: compute_outgoing_fluxes on one patch of full U_sub
print()
print("=" * 60)
print("Test 4: compute_outgoing_fluxes on selected patch")
print("=" * 60)

# Reuse deterministic field above so neighbour values are available.

flux = compute_outgoing_fluxes(
    U_test,
    dx=dx,
    interior_method="central",
    boundary_method="central_zero",
)

neigh_rt = extract_neighbours_indexes(U_test)

# Pick a patch that always exists and has a right neighbour when possible.
# With 3x3 elements per patch, the grid is 3 x 2 patches, so ix=0 is the only
# column with a right neighbour.
iy = 1 if n_suby > 2 else 0
ix = 0
bottom = flux["bottom"][iy, ix, :, 0]
top = flux["top"][iy, ix, :, 0]
left = flux["left"][iy, ix, :, 0]
right = flux["right"][iy, ix, :, 0]

# Expected values for this selected patch.
# Flux arrays exclude corner entries, so compare only interior face nodes.
face_cols = slice(1, cn - 1)
face_rows = slice(1, rn - 1)

has_bottom = neigh_rt[iy, ix]["bottom"] != -1
has_right = neigh_rt[iy, ix]["right"] != -1
has_top = neigh_rt[iy, ix]["top"] != -1
has_left = neigh_rt[iy, ix]["left"] != -1

if has_bottom:
    ny_bottom, nx_bottom = iy - 1, ix
    expected_bottom = (U_test[ny_bottom, nx_bottom, rn - 2, face_cols, 0] - U_test[iy, ix, 1, face_cols, 0]) / (2 * dx)
else:
    expected_bottom = -U_test[iy, ix, 1, face_cols, 0] / (2 * dx)

if has_top:
    ny_top, nx_top = iy + 1, ix
    expected_top = (U_test[ny_top, nx_top, 1, face_cols, 0] - U_test[iy, ix, rn - 2, face_cols, 0]) / (2 * dx)
else:
    expected_top = -U_test[iy, ix, rn - 2, face_cols, 0] / (2 * dx)

if has_left:
    ny_left, nx_left = iy, ix - 1
    expected_left = (U_test[ny_left, nx_left, face_rows, cn - 2, 0] - U_test[iy, ix, face_rows, 1, 0]) / (2 * dx)
else:
    expected_left = -U_test[iy, ix, face_rows, 1, 0] / (2 * dx)

if has_right:
    ny_right, nx_right = iy, ix + 1
    expected_right = (U_test[ny_right, nx_right, face_rows, 1, 0] - U_test[iy, ix, face_rows, cn - 2, 0]) / (2 * dx)
else:
    expected_right = -U_test[iy, ix, face_rows, cn - 2, 0] / (2 * dx)

print(f"Selected patch: (iy={iy}, ix={ix})")
print("Selected patch nodal values (top row first):")
print(np.flipud(U_test[iy, ix, :, :, 0]))
print()
print("Computed bottom flux:", bottom)
print("Expected bottom flux:", expected_bottom)
print("Computed top flux   :", top)
print("Expected top flux   :", expected_top)
print("Computed left flux  :", left)
print("Expected left flux  :", expected_left)
print("Computed right flux :", right)
print("Expected right flux :", expected_right)

ok = (
    np.allclose(bottom, expected_bottom)
    and np.allclose(top, expected_top)
    and np.allclose(left, expected_left)
    and np.allclose(right, expected_right)
)
print()
print("External boundary flux test passed:", ok)
if not ok:
    raise AssertionError("compute_outgoing_fluxes external boundary test failed")


# Test 5: modify_subdomain_boundary_values and restore_subdomain_boundary_values round-trip
print()
print("=" * 60)
print("Test 5: modify and restore subdomain boundary values (round-trip)")
print("=" * 60)

# Use multiple samples to exercise vectorization
n_samples_rt = 3
dx_rt = 1.5
dy_rt = 2.0
U0 = np.zeros((n_suby, n_subx, rn, cn, n_samples_rt), dtype=float)
for s in range(n_samples_rt):
    for iy in range(n_suby):
        for ix in range(n_subx):
            for r in range(rn):
                for c in range(cn):
                    U0[iy, ix, r, c, s] = (1000.0 * iy + 100.0 * ix + 10.0 * r + c) + 1.0 * s

Umod = U0.copy()

# Apply modification
modify_subdomain_boundary_values(Umod, dx_rt, dy_rt)

# Verify that interior face nodes were reduced by the linear interpolation
faces_rt, corners_rt = subelement_boundary_local_coords(nsely, nselx)
ok_mod = True
for iy in range(n_suby):
    for ix in range(n_subx):
        bl = U0[iy, ix, corners_rt['bottom_left'][0], corners_rt['bottom_left'][1], :]
        br = U0[iy, ix, corners_rt['bottom_right'][0], corners_rt['bottom_right'][1], :]
        tl = U0[iy, ix, corners_rt['top_left'][0], corners_rt['top_left'][1], :]
        tr = U0[iy, ix, corners_rt['top_right'][0], corners_rt['top_right'][1], :]

        # horizontal faces
        if cn > 1 and dx_rt != 0:
            denom_x_phys = (cn - 1) * float(dx_rt)
            for c in range(1, cn - 1):
                frac = (c * float(dx_rt)) / denom_x_phys
                linear_bottom = bl + frac * (br - bl)
                expected = U0[iy, ix, 0, c, :] - linear_bottom
                if not np.allclose(Umod[iy, ix, 0, c, :], expected):
                    ok_mod = False
                linear_top = tl + frac * (tr - tl)
                expected_t = U0[iy, ix, rn - 1, c, :] - linear_top
                if not np.allclose(Umod[iy, ix, rn - 1, c, :], expected_t):
                    ok_mod = False

        # vertical faces
        if rn > 1 and dy_rt != 0:
            denom_y_phys = (rn - 1) * float(dy_rt)
            for r in range(1, rn - 1):
                frac = (r * float(dy_rt)) / denom_y_phys
                linear_left = bl + frac * (tl - bl)
                expected_l = U0[iy, ix, r, 0, :] - linear_left
                if not np.allclose(Umod[iy, ix, r, 0, :], expected_l):
                    ok_mod = False
                linear_right = br + frac * (tr - br)
                expected_r = U0[iy, ix, r, cn - 1, :] - linear_right
                if not np.allclose(Umod[iy, ix, r, cn - 1, :], expected_r):
                    ok_mod = False

print("Modification applied correctly:", ok_mod)
if not ok_mod:
    raise AssertionError("modify_subdomain_boundary_values did not produce expected values")

# Restore and verify exact round-trip
restore_subdomain_boundary_values(Umod, dx_rt, dy_rt)
ok_restore = np.allclose(Umod, U0)
print("Restore round-trip successful:", ok_restore)
if not ok_restore:
    raise AssertionError("restore_subdomain_boundary_values failed to restore original U_sub")
