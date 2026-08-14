import numpy as np

def subelement_boundary_local_coords(nsely, nselx):
    """Return patch-local node (row,col) coords for each boundary side.

    - nsely,nselx: elements per patch in y,x
    - node grid inside patch has shape (nsely+1, nselx+1)
    Returns dict with lists of (row,col) coords (ordered along the face).
    """
    rn = nsely + 1
    cn = nselx + 1
    bottom = [(0, c) for c in range(1, cn - 1)]
    top    = [(rn - 1, c) for c in range(1, cn - 1)]
    left   = [(r, 0) for r in range(1, rn - 1)]
    right  = [(r, cn - 1) for r in range(1, rn - 1)]
    bottom_left = (0, 0)
    bottom_right = (0, cn - 1)  
    top_left = (rn - 1, 0)
    top_right = (rn - 1, cn - 1)
    faces = {"bottom": bottom, "right": right, "top": top, "left": left}
    corners = {"bottom_left": bottom_left, "bottom_right": bottom_right, "top_left": top_left, "top_right": top_right}
    return faces, corners



def extract_neighbours_indexes(U_sub):
    """Return neighbour patch indices for each patch in the tiling.

    Parameters
    - U_sub: patch arrays with shape
      (n_suby, n_subx, nsely+1, nselx+1, n_samples)
    - nely, nelx: total number of elements in y and x (global) [unused]
    - nselx, nsely: number of elements per patch in x and y [unused]

    Returns
    - neighbours: structured array shaped (n_suby, n_subx) with fields
      'bottom', 'right', 'top', 'left'. Each field contains the neighbour
      patch index as a linear ID (iy*n_subx + ix), or -1 if that neighbour
      patch does not exist (domain boundary).

    Example
    -------
    neighbours[iy, ix]['bottom']  # get bottom neighbour of patch (iy, ix)
    """
    n_suby, n_subx = U_sub.shape[:2]
    
    # Create structured array with named fields
    dtype = [('bottom', int), ('right', int), ('top', int), ('left', int)]
    neighbours = np.zeros((n_suby, n_subx), dtype=dtype)

    for iy in range(n_suby):
        for ix in range(n_subx):
            # bottom: iy-1, ix
            if iy - 1 >= 0:
                neighbours[iy, ix]['bottom'] = (iy - 1) * n_subx + ix
            else:
                neighbours[iy, ix]['bottom'] = -1

            # right: iy, ix+1
            if ix + 1 < n_subx:
                neighbours[iy, ix]['right'] = iy * n_subx + (ix + 1)
            else:
                neighbours[iy, ix]['right'] = -1

            # top: iy+1, ix
            if iy + 1 < n_suby:
                neighbours[iy, ix]['top'] = (iy + 1) * n_subx + ix
            else:
                neighbours[iy, ix]['top'] = -1

            # left: iy, ix-1
            if ix - 1 >= 0:
                neighbours[iy, ix]['left'] = iy * n_subx + (ix - 1)
            else:
                neighbours[iy, ix]['left'] = -1

    return neighbours


def compute_outgoing_fluxes(
    U_sub,
    dx,
    interior_method="central",
    boundary_method="central_zero",
):
    """Compute outgoing normal flux on patch faces for each `U_sub`.

    Parameters
    - U_sub: array shaped (n_suby, n_subx, nsely+1, nselx+1, n_samples)
    - dx: grid spacing (assumed same in x and y)
    - interior_method: scheme for faces that have a neighbour patch
      * "central"
      * "inside_first"
      * "inside_second"
      * "outside_first"
      * "outside_second"
    - boundary_method: scheme for outer domain boundary faces
      * "central_zero"
      * "inside_first"
      * "inside_second"

    Returns
    - fluxes: dict with keys 'bottom', 'right', 'top', 'left'.
      Each entry stores the outgoing normal derivative on that face.
            Corner entries are not included in the returned arrays.
            Shapes are (n_suby, n_subx, nselx-1, n_samples) for bottom/top and
            (n_suby, n_subx, nsely-1, n_samples) for left/right.

    Notes
    - "central" uses the two values on both sides of the face.
    - "inside_*" uses values from the patch interior side only.
    - "outside_*" uses values from the neighbour side only.
    - "central_zero" treats the missing outside value as 0.
    """
    n_suby, n_subx = U_sub.shape[:2]
    rn = U_sub.shape[2]
    cn = U_sub.shape[3]
    n_samples = U_sub.shape[4]

    neigh = extract_neighbours_indexes(U_sub)

    n_face_x = max(cn - 2, 0)
    n_face_y = max(rn - 2, 0)
    bottom_flux = np.zeros((n_suby, n_subx, n_face_x, n_samples), dtype=float)
    top_flux = np.zeros((n_suby, n_subx, n_face_x, n_samples), dtype=float)
    left_flux = np.zeros((n_suby, n_subx, n_face_y, n_samples), dtype=float)
    right_flux = np.zeros((n_suby, n_subx, n_face_y, n_samples), dtype=float)

    def _central(face_in, face_out, axis_sign):
        return axis_sign * (face_out - face_in) / (2 * dx)

    def _inside_first(face, inner, axis_sign):
        return axis_sign * (face - inner) / dx

    def _inside_second(face, inner1, inner2, axis_sign):
        return axis_sign * (3 * face - 4 * inner1 + inner2) / (2 * dx)

    def _outside_first(face, outer, axis_sign):
        return axis_sign * (outer - face) / dx

    def _outside_second(face, outer1, outer2, axis_sign):
        return axis_sign * (-3 * face + 4 * outer1 - outer2) / (2 * dx)

    def _boundary_zero(face_in, axis_sign):
        return axis_sign * (0.0 - face_in) / (2 * dx)

    for iy in range(n_suby):
        for ix in range(n_subx):
            has_bottom = neigh[iy, ix]['bottom'] != -1
            has_right = neigh[iy, ix]['right'] != -1
            has_top = neigh[iy, ix]['top'] != -1
            has_left = neigh[iy, ix]['left'] != -1

            # Skip corners; compute and store face fluxes only on interior face nodes.
            for c_out, c in enumerate(range(1, cn - 1)):
                if has_bottom:
                    nb = neigh[iy, ix]['bottom']
                    ny = nb // n_subx
                    nx = nb % n_subx
                    if interior_method == "central":
                        dU_dy = _central(U_sub[iy, ix, 1, c, :], U_sub[ny, nx, rn - 2, c, :], +1)
                    elif interior_method == "inside_first":
                        dU_dy = _inside_first(U_sub[iy, ix, 0, c, :], U_sub[iy, ix, 1, c, :], +1)
                    elif interior_method == "inside_second":
                        dU_dy = _inside_second(U_sub[iy, ix, 0, c, :], U_sub[iy, ix, 1, c, :], U_sub[iy, ix, 2, c, :], +1)
                    elif interior_method == "outside_first":
                        dU_dy = _outside_first(U_sub[iy, ix, 0, c, :], U_sub[ny, nx, rn - 2, c, :], +1)
                    elif interior_method == "outside_second":
                        dU_dy = _outside_second(U_sub[iy, ix, 0, c, :], U_sub[ny, nx, rn - 2, c, :], U_sub[ny, nx, rn - 3, c, :], +1)
                    else:
                        raise ValueError(f"Unknown interior_method: {interior_method}")
                else:
                    if boundary_method == "central_zero":
                        dU_dy = _boundary_zero(U_sub[iy, ix, 1, c, :], +1)
                    elif boundary_method == "inside_first":
                        dU_dy = _inside_first(U_sub[iy, ix, 0, c, :], U_sub[iy, ix, 1, c, :], +1)
                    elif boundary_method == "inside_second":
                        dU_dy = _inside_second(U_sub[iy, ix, 0, c, :], U_sub[iy, ix, 1, c, :], U_sub[iy, ix, 2, c, :], +1)
                    else:
                        raise ValueError(f"Unknown boundary_method: {boundary_method}")
                bottom_flux[iy, ix, c_out, :] = dU_dy

                if has_top:
                    nt = neigh[iy, ix]['top']
                    ny = nt // n_subx
                    nx = nt % n_subx
                    if interior_method == "central":
                        dU_dy = _central(U_sub[iy, ix, rn - 2, c, :], U_sub[ny, nx, 1, c, :], +1)
                    elif interior_method == "inside_first":
                        dU_dy = _inside_first(U_sub[iy, ix, rn - 1, c, :], U_sub[iy, ix, rn - 2, c, :], +1)
                    elif interior_method == "inside_second":
                        dU_dy = _inside_second(U_sub[iy, ix, rn - 1, c, :], U_sub[iy, ix, rn - 2, c, :], U_sub[iy, ix, rn - 3, c, :], +1)
                    elif interior_method == "outside_first":
                        dU_dy = _outside_first(U_sub[iy, ix, rn - 1, c, :], U_sub[ny, nx, 1, c, :], +1)
                    elif interior_method == "outside_second":
                        dU_dy = _outside_second(U_sub[iy, ix, rn - 1, c, :], U_sub[ny, nx, 1, c, :], U_sub[ny, nx, 2, c, :], +1)
                    else:
                        raise ValueError(f"Unknown interior_method: {interior_method}")
                else:
                    if boundary_method == "central_zero":
                        dU_dy = _boundary_zero(U_sub[iy, ix, rn - 2, c, :], +1)
                    elif boundary_method == "inside_first":
                        dU_dy = _inside_first(U_sub[iy, ix, rn - 1, c, :], U_sub[iy, ix, rn - 2, c, :], +1)
                    elif boundary_method == "inside_second":
                        dU_dy = _inside_second(U_sub[iy, ix, rn - 1, c, :], U_sub[iy, ix, rn - 2, c, :], U_sub[iy, ix, rn - 3, c, :], +1)
                    else:
                        raise ValueError(f"Unknown boundary_method: {boundary_method}")
                top_flux[iy, ix, c_out, :] = dU_dy

            for r_out, r in enumerate(range(1, rn - 1)):
                if has_left:
                    nl = neigh[iy, ix]['left']
                    ny = nl // n_subx
                    nx = nl % n_subx
                    if interior_method == "central":
                        dU_dx = _central(U_sub[iy, ix, r, 1, :], U_sub[ny, nx, r, cn - 2, :], +1)
                    elif interior_method == "inside_first":
                        dU_dx = _inside_first(U_sub[iy, ix, r, 0, :], U_sub[iy, ix, r, 1, :], +1)
                    elif interior_method == "inside_second":
                        dU_dx = _inside_second(U_sub[iy, ix, r, 0, :], U_sub[iy, ix, r, 1, :], U_sub[iy, ix, r, 2, :], +1)
                    elif interior_method == "outside_first":
                        dU_dx = _outside_first(U_sub[iy, ix, r, 0, :], U_sub[ny, nx, r, cn - 2, :], +1)
                    elif interior_method == "outside_second":
                        dU_dx = _outside_second(U_sub[iy, ix, r, 0, :], U_sub[ny, nx, r, cn - 2, :], U_sub[ny, nx, r, cn - 3, :], +1)
                    else:
                        raise ValueError(f"Unknown interior_method: {interior_method}")
                else:
                    if boundary_method == "central_zero":
                        dU_dx = _boundary_zero(U_sub[iy, ix, r, 1, :], +1)
                    elif boundary_method == "inside_first":
                        dU_dx = _inside_first(U_sub[iy, ix, r, 0, :], U_sub[iy, ix, r, 1, :], +1)
                    elif boundary_method == "inside_second":
                        dU_dx = _inside_second(U_sub[iy, ix, r, 0, :], U_sub[iy, ix, r, 1, :], U_sub[iy, ix, r, 2, :], +1)
                    else:
                        raise ValueError(f"Unknown boundary_method: {boundary_method}")
                left_flux[iy, ix, r_out, :] = dU_dx

                if has_right:
                    nr = neigh[iy, ix]['right']
                    ny = nr // n_subx
                    nx = nr % n_subx
                    if interior_method == "central":
                        dU_dx = _central(U_sub[iy, ix, r, cn - 2, :], U_sub[ny, nx, r, 1, :], +1)
                    elif interior_method == "inside_first":
                        dU_dx = _inside_first(U_sub[iy, ix, r, cn - 1, :], U_sub[iy, ix, r, cn - 2, :], +1)
                    elif interior_method == "inside_second":
                        dU_dx = _inside_second(U_sub[iy, ix, r, cn - 1, :], U_sub[iy, ix, r, cn - 2, :], U_sub[iy, ix, r, cn - 3, :], +1)
                    elif interior_method == "outside_first":
                        dU_dx = _outside_first(U_sub[iy, ix, r, cn - 1, :], U_sub[ny, nx, r, 1, :], +1)
                    elif interior_method == "outside_second":
                        dU_dx = _outside_second(U_sub[iy, ix, r, cn - 1, :], U_sub[ny, nx, r, 1, :], U_sub[ny, nx, r, 2, :], +1)
                    else:
                        raise ValueError(f"Unknown interior_method: {interior_method}")
                else:
                    if boundary_method == "central_zero":
                        dU_dx = _boundary_zero(U_sub[iy, ix, r, cn - 2, :], +1)
                    elif boundary_method == "inside_first":
                        dU_dx = _inside_first(U_sub[iy, ix, r, cn - 1, :], U_sub[iy, ix, r, cn - 2, :], +1)
                    elif boundary_method == "inside_second":
                        dU_dx = _inside_second(U_sub[iy, ix, r, cn - 1, :], U_sub[iy, ix, r, cn - 2, :], U_sub[iy, ix, r, cn - 3, :], +1)
                    else:
                        raise ValueError(f"Unknown boundary_method: {boundary_method}")
                right_flux[iy, ix, r_out, :] = dU_dx

    return {"bottom": bottom_flux, "right": right_flux, "top": top_flux, "left": left_flux}


def compute_outgoing_fluxes_with_corners(
    U_sub,
    dx,
    interior_method="central",
    boundary_method="central_zero",
):
    """Compute outgoing normal flux on patch faces including corners for each `U_sub`.

    Parameters
    - U_sub: array shaped (n_suby, n_subx, nsely+1, nselx+1, n_samples)
    - dx: grid spacing (assumed same in x and y)
    - interior_method: scheme for faces that have a neighbour patch
      * "central"
      * "inside_first"
      * "inside_second"
      * "outside_first"
      * "outside_second"
    - boundary_method: scheme for outer domain boundary faces
      * "central_zero"
      * "inside_first"
      * "inside_second"

    Returns
    - fluxes: dict with keys 'bottom', 'right', 'top', 'left'.
      Each entry stores the outgoing normal derivative on that face including corners.
      Shapes are (n_suby, n_subx, nselx+1, n_samples) for bottom/top and
      (n_suby, n_subx, nsely+1, n_samples) for left/right.
    """
    n_suby, n_subx = U_sub.shape[:2]
    rn = U_sub.shape[2]  # nsely + 1
    cn = U_sub.shape[3]  # nselx + 1
    n_samples = U_sub.shape[4]

    neigh = extract_neighbours_indexes(U_sub)

    # Face lengths now include the corners (cn for x-faces, rn for y-faces)
    bottom_flux = np.zeros((n_suby, n_subx, cn, n_samples), dtype=float)
    top_flux = np.zeros((n_suby, n_subx, cn, n_samples), dtype=float)
    left_flux = np.zeros((n_suby, n_subx, rn, n_samples), dtype=float)
    right_flux = np.zeros((n_suby, n_subx, rn, n_samples), dtype=float)

    def _central(face_in, face_out, axis_sign):
        return axis_sign * (face_out - face_in) / (2 * dx)

    def _inside_first(face, inner, axis_sign):
        return axis_sign * (face - inner) / dx

    def _inside_second(face, inner1, inner2, axis_sign):
        return axis_sign * (3 * face - 4 * inner1 + inner2) / (2 * dx)

    def _outside_first(face, outer, axis_sign):
        return axis_sign * (outer - face) / dx

    def _outside_second(face, outer1, outer2, axis_sign):
        return axis_sign * (-3 * face + 4 * outer1 - outer2) / (2 * dx)

    def _boundary_zero(face_in, axis_sign):
        return axis_sign * (0.0 - face_in) / (2 * dx)

    for iy in range(n_suby):
        for ix in range(n_subx):
            has_bottom = neigh[iy, ix]['bottom'] != -1
            has_right = neigh[iy, ix]['right'] != -1
            has_top = neigh[iy, ix]['top'] != -1
            has_left = neigh[iy, ix]['left'] != -1

            # Iterate through all columns (0 to cn - 1) including corners
            for c in range(cn):
                # Bottom face
                if has_bottom:
                    nb = neigh[iy, ix]['bottom']
                    ny, nx = nb // n_subx, nb % n_subx
                    if interior_method == "central":
                        dU_dy = _central(U_sub[iy, ix, 1, c, :], U_sub[ny, nx, rn - 2, c, :], +1)
                    elif interior_method == "inside_first":
                        dU_dy = _inside_first(U_sub[iy, ix, 0, c, :], U_sub[iy, ix, 1, c, :], +1)
                    elif interior_method == "inside_second":
                        dU_dy = _inside_second(U_sub[iy, ix, 0, c, :], U_sub[iy, ix, 1, c, :], U_sub[iy, ix, 2, c, :], +1)
                    elif interior_method == "outside_first":
                        dU_dy = _outside_first(U_sub[iy, ix, 0, c, :], U_sub[ny, nx, rn - 2, c, :], +1)
                    elif interior_method == "outside_second":
                        dU_dy = _outside_second(U_sub[iy, ix, 0, c, :], U_sub[ny, nx, rn - 2, c, :], U_sub[ny, nx, rn - 3, c, :], +1)
                    else:
                        raise ValueError(f"Unknown interior_method: {interior_method}")
                else:
                    if boundary_method == "central_zero":
                        dU_dy = _boundary_zero(U_sub[iy, ix, 1, c, :], +1)
                    elif boundary_method == "inside_first":
                        dU_dy = _inside_first(U_sub[iy, ix, 0, c, :], U_sub[iy, ix, 1, c, :], +1)
                    elif boundary_method == "inside_second":
                        dU_dy = _inside_second(U_sub[iy, ix, 0, c, :], U_sub[iy, ix, 1, c, :], U_sub[iy, ix, 2, c, :], +1)
                    else:
                        raise ValueError(f"Unknown boundary_method: {boundary_method}")
                bottom_flux[iy, ix, c, :] = dU_dy

                # Top face
                if has_top:
                    nt = neigh[iy, ix]['top']
                    ny, nx = nt // n_subx, nt % n_subx
                    if interior_method == "central":
                        dU_dy = _central(U_sub[iy, ix, rn - 2, c, :], U_sub[ny, nx, 1, c, :], +1)
                    elif interior_method == "inside_first":
                        dU_dy = _inside_first(U_sub[iy, ix, rn - 1, c, :], U_sub[iy, ix, rn - 2, c, :], +1)
                    elif interior_method == "inside_second":
                        dU_dy = _inside_second(U_sub[iy, ix, rn - 1, c, :], U_sub[iy, ix, rn - 2, c, :], U_sub[iy, ix, rn - 3, c, :], +1)
                    elif interior_method == "outside_first":
                        dU_dy = _outside_first(U_sub[iy, ix, rn - 1, c, :], U_sub[ny, nx, 1, c, :], +1)
                    elif interior_method == "outside_second":
                        dU_dy = _outside_second(U_sub[iy, ix, rn - 1, c, :], U_sub[ny, nx, 1, c, :], U_sub[ny, nx, 2, c, :], +1)
                    else:
                        raise ValueError(f"Unknown interior_method: {interior_method}")
                else:
                    if boundary_method == "central_zero":
                        dU_dy = _boundary_zero(U_sub[iy, ix, rn - 2, c, :], +1)
                    elif boundary_method == "inside_first":
                        dU_dy = _inside_first(U_sub[iy, ix, rn - 1, c, :], U_sub[iy, ix, rn - 2, c, :], +1)
                    elif boundary_method == "inside_second":
                        dU_dy = _inside_second(U_sub[iy, ix, rn - 1, c, :], U_sub[iy, ix, rn - 2, c, :], U_sub[iy, ix, rn - 3, c, :], +1)
                    else:
                        raise ValueError(f"Unknown boundary_method: {boundary_method}")
                top_flux[iy, ix, c, :] = dU_dy

            # Iterate through all rows (0 to rn - 1) including corners
            for r in range(rn):
                # Left face
                if has_left:
                    nl = neigh[iy, ix]['left']
                    ny, nx = nl // n_subx, nl % n_subx
                    if interior_method == "central":
                        dU_dx = _central(U_sub[iy, ix, r, 1, :], U_sub[ny, nx, r, cn - 2, :], +1)
                    elif interior_method == "inside_first":
                        dU_dx = _inside_first(U_sub[iy, ix, r, 0, :], U_sub[iy, ix, r, 1, :], +1)
                    elif interior_method == "inside_second":
                        dU_dx = _inside_second(U_sub[iy, ix, r, 0, :], U_sub[iy, ix, r, 1, :], U_sub[iy, ix, r, 2, :], +1)
                    elif interior_method == "outside_first":
                        dU_dx = _outside_first(U_sub[iy, ix, r, 0, :], U_sub[ny, nx, r, cn - 2, :], +1)
                    elif interior_method == "outside_second":
                        dU_dx = _outside_second(U_sub[iy, ix, r, 0, :], U_sub[ny, nx, r, cn - 2, :], U_sub[ny, nx, r, cn - 3, :], +1)
                    else:
                        raise ValueError(f"Unknown interior_method: {interior_method}")
                else:
                    if boundary_method == "central_zero":
                        dU_dx = _boundary_zero(U_sub[iy, ix, r, 1, :], +1)
                    elif boundary_method == "inside_first":
                        dU_dx = _inside_first(U_sub[iy, ix, r, 0, :], U_sub[iy, ix, r, 1, :], +1)
                    elif boundary_method == "inside_second":
                        dU_dx = _inside_second(U_sub[iy, ix, r, 0, :], U_sub[iy, ix, r, 1, :], U_sub[iy, ix, r, 2, :], +1)
                    else:
                        raise ValueError(f"Unknown boundary_method: {boundary_method}")
                left_flux[iy, ix, r, :] = dU_dx

                # Right face
                if has_right:
                    nr = neigh[iy, ix]['right']
                    ny, nx = nr // n_subx, nr % n_subx
                    if interior_method == "central":
                        dU_dx = _central(U_sub[iy, ix, r, cn - 2, :], U_sub[ny, nx, r, 1, :], +1)
                    elif interior_method == "inside_first":
                        dU_dx = _inside_first(U_sub[iy, ix, r, cn - 1, :], U_sub[iy, ix, r, cn - 2, :], +1)
                    elif interior_method == "inside_second":
                        dU_dx = _inside_second(U_sub[iy, ix, r, cn - 1, :], U_sub[iy, ix, r, cn - 2, :], U_sub[iy, ix, r, cn - 3, :], +1)
                    elif interior_method == "outside_first":
                        dU_dx = _outside_first(U_sub[iy, ix, r, cn - 1, :], U_sub[ny, nx, r, 1, :], +1)
                    elif interior_method == "outside_second":
                        dU_dx = _outside_second(U_sub[iy, ix, r, cn - 1, :], U_sub[ny, nx, r, 1, :], U_sub[ny, nx, r, 2, :], +1)
                    else:
                        raise ValueError(f"Unknown interior_method: {interior_method}")
                else:
                    if boundary_method == "central_zero":
                        dU_dx = _boundary_zero(U_sub[iy, ix, r, cn - 2, :], +1)
                    elif boundary_method == "inside_first":
                        dU_dx = _inside_first(U_sub[iy, ix, r, cn - 1, :], U_sub[iy, ix, r, cn - 2, :], +1)
                    elif boundary_method == "inside_second":
                        dU_dx = _inside_second(U_sub[iy, ix, r, cn - 1, :], U_sub[iy, ix, r, cn - 2, :], U_sub[iy, ix, r, cn - 3, :], +1)
                    else:
                        raise ValueError(f"Unknown boundary_method: {boundary_method}")
                right_flux[iy, ix, r, :] = dU_dx

    return {"bottom": bottom_flux, "right": right_flux, "top": top_flux, "left": left_flux}

def modify_subdomain_boundary_values(U_sub,dx,dy):
    """Return modified `U_sub` with specified values on patch boundaries.

    Parameters
    - U_sub: array shaped (n_suby, n_subx, nsely+1, nselx+1, n_samples)
    - boundary_values: dict with keys 'bottom', 'right', 'top', 'left'.
      Each entry is an array of shape (n_suby, n_subx, face_length, n_samples)
      containing the new values to set on that face. `face_length` should be
      nselx-1 for bottom/top and nsely-1 for left/right.

    Returns
    - U_mod: modified copy of `U_sub` with new boundary values.
    """
    n_suby, n_subx = U_sub.shape[:2]
    rn = U_sub.shape[2]
    cn = U_sub.shape[3]
    n_samples = U_sub.shape[4]

    # faces: interior face node coordinates (excluding corners)
    faces, corners = subelement_boundary_local_coords(rn - 1, cn - 1)

    # Modify U_sub in-place: remove linear interpolation between corners
    # along each face for every patch and sample. Use DX and DY as the
    # physical spacing between adjacent nodes in x and y so the linear
    # interpolation is computed in physical coordinates.
    for iy in range(n_suby):
        for ix in range(n_subx):
            # corner values (shape: (n_samples,))
            bl = U_sub[iy, ix, corners['bottom_left'][0], corners['bottom_left'][1], :]
            br = U_sub[iy, ix, corners['bottom_right'][0], corners['bottom_right'][1], :]
            tl = U_sub[iy, ix, corners['top_left'][0], corners['top_left'][1], :]
            tr = U_sub[iy, ix, corners['top_right'][0], corners['top_right'][1], :]

            # horizontal faces (bottom/top): interpolate between left and right corners
            if cn > 1 and dx != 0:
                denom_x_phys = (cn - 1) * float(dx)
                for c in range(1, cn - 1):
                    frac = (c * float(dx)) / denom_x_phys
                    # bottom face node at (0,c)
                    linear_bottom = bl + frac * (br - bl)
                    U_sub[iy, ix, 0, c, :] = U_sub[iy, ix, 0, c, :] - linear_bottom
                    # top face node at (rn-1,c)
                    linear_top = tl + frac * (tr - tl)
                    U_sub[iy, ix, rn - 1, c, :] = U_sub[iy, ix, rn - 1, c, :] - linear_top

            # vertical faces (left/right): interpolate between bottom and top corners
            if rn > 1 and dy != 0:
                denom_y_phys = (rn - 1) * float(dy)
                for r in range(1, rn - 1):
                    frac = (r * float(dy)) / denom_y_phys
                    # left face node at (r,0)
                    linear_left = bl + frac * (tl - bl)
                    U_sub[iy, ix, r, 0, :] = U_sub[iy, ix, r, 0, :] - linear_left
                    # right face node at (r,cn-1)
                    linear_right = br + frac * (tr - br)
                    U_sub[iy, ix, r, cn - 1, :] = U_sub[iy, ix, r, cn - 1, :] - linear_right

    return U_sub


def restore_subdomain_boundary_values(U_sub, dx, dy):
    """Inverse of `modify_subdomain_boundary_values`.

    Adds back the linear component along each face of every patch.

    Parameters
    - U_sub: array shaped (n_suby, n_subx, nsely+1, nselx+1, n_samples)
    - dx, dy: physical spacing between adjacent nodes in x and y

    Modifies `U_sub` in-place and returns it.
    """
    n_suby, n_subx = U_sub.shape[:2]
    rn = U_sub.shape[2]
    cn = U_sub.shape[3]

    faces, corners = subelement_boundary_local_coords(rn - 1, cn - 1)

    for iy in range(n_suby):
        for ix in range(n_subx):
            bl = U_sub[iy, ix, corners['bottom_left'][0], corners['bottom_left'][1], :]
            br = U_sub[iy, ix, corners['bottom_right'][0], corners['bottom_right'][1], :]
            tl = U_sub[iy, ix, corners['top_left'][0], corners['top_left'][1], :]
            tr = U_sub[iy, ix, corners['top_right'][0], corners['top_right'][1], :]

            # horizontal faces: bottom/top
            if cn > 1 and dx != 0:
                denom_x_phys = (cn - 1) * float(dx)
                for c in range(1, cn - 1):
                    frac = (c * float(dx)) / denom_x_phys
                    linear_bottom = bl + frac * (br - bl)
                    U_sub[iy, ix, 0, c, :] = U_sub[iy, ix, 0, c, :] + linear_bottom
                    linear_top = tl + frac * (tr - tl)
                    U_sub[iy, ix, rn - 1, c, :] = U_sub[iy, ix, rn - 1, c, :] + linear_top

            # vertical faces: left/right
            if rn > 1 and dy != 0:
                denom_y_phys = (rn - 1) * float(dy)
                for r in range(1, rn - 1):
                    frac = (r * float(dy)) / denom_y_phys
                    linear_left = bl + frac * (tl - bl)
                    U_sub[iy, ix, r, 0, :] = U_sub[iy, ix, r, 0, :] + linear_left
                    linear_right = br + frac * (tr - br)
                    U_sub[iy, ix, r, cn - 1, :] = U_sub[iy, ix, r, cn - 1, :] + linear_right

    return U_sub

def extract_boundary_values(U_sub):
    """Extract boundary values from `U_sub` for each patch and face.

    Parameters
    - U_sub: array shaped (n_suby, n_subx, nsely+1, nselx+1, n_samples)

    Returns
    - boundary_values: dict with keys 'bottom', 'right', 'top', 'left'.
      Each entry is an array of shape (n_suby, n_subx, face_length, n_samples)
      containing the values on that face. `face_length` is nselx-1 for bottom/top
      and nsely-1 for left/right. Corner nodes are not included in the returned arrays.
    """
    n_suby, n_subx = U_sub.shape[:2]
    rn = U_sub.shape[2]
    cn = U_sub.shape[3]

    faces, corners = subelement_boundary_local_coords(rn - 1, cn - 1)

    boundary_values = {}
    for face_name, coords in faces.items():
        if face_name in ['bottom', 'top']:
            face_length = cn - 2
        else:
            face_length = rn - 2

        values = np.zeros((n_suby, n_subx, face_length, U_sub.shape[4]), dtype=U_sub.dtype)
        for i, (r, c) in enumerate(coords):
            values[:, :, i, :] = U_sub[:, :, r, c, :]
        boundary_values[face_name] = values

    return boundary_values


def extract_corner_values(U_sub):
    """Extract corner values from `U_sub` for each patch.

    Parameters
    - U_sub: array shaped (n_suby, n_subx, nsely+1, nselx+1, n_samples)

    Returns
    - corner_values: dict with keys 'bottom_left', 'bottom_right', 'top_left', 'top_right'.
      Each entry is an array of shape (n_suby, n_subx, n_samples) containing the value
      at that patch corner.
    """
    n_suby, n_subx = U_sub.shape[:2]
    rn = U_sub.shape[2]
    cn = U_sub.shape[3]

    _, corners = subelement_boundary_local_coords(rn - 1, cn - 1)

    corner_values = {}
    for corner_name, (r, c) in corners.items():
        corner_values[corner_name] = U_sub[:, :, r, c, :]

    return corner_values


