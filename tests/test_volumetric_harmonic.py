import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import Delaunay

# ---------------- FEM assembly (volumetric Laplacian) ----------------

def assemble_tet_laplacian(V, T):
    """
    Linear FEM stiffness matrix for Laplace operator on a tetrahedral mesh.
    V: (nv,3) float64 vertices
    T: (mt,4) int32 tetrahedra (vertex indices)
    Returns: K (nv x nv) csr_matrix
    """
    nv = V.shape[0]
    I = []; J = []; S = []

    for tet in T:
        i0, i1, i2, i3 = tet
        v0, v1, v2, v3 = V[i0], V[i1], V[i2], V[i3]
        X = np.column_stack((v1 - v0, v2 - v0, v3 - v0))  # 3x3
        detX = np.linalg.det(X)
        vol = abs(detX) / 6.0
        if vol <= 1e-16:
            continue
        invXT = np.linalg.inv(X).T
        g1 = invXT[:,0]; g2 = invXT[:,1]; g3 = invXT[:,2]
        g0 = -(g1 + g2 + g3)
        grads = np.stack([g0,g1,g2,g3], axis=0)  # (4,3)
        Ke = vol * (grads @ grads.T)            # (4,4)

        idx = [i0,i1,i2,i3]
        for a in range(4):
            ia = idx[a]
            for b in range(4):
                ib = idx[b]
                I.append(ia); J.append(ib); S.append(Ke[a,b])

    K = coo_matrix((S,(I,J)), shape=(nv,nv)).tocsr()
    return K

# ---------------- Utilities ----------------

def simple_tetrahedralize_from_surface(V_surf, F_surf, extra_points=None):
    """
    Crude tetrahedralization via 3D Delaunay on the surface vertices (plus optional extra interior points).
    Works well if the shape is close to convex. For non-convex shapes, replace with a proper tet mesher.

    Returns:
      V_vol (nv,3), T (mt,4), surf_map (np.array of indices mapping each surface vertex to same index in V_vol)
    """
    if extra_points is not None and len(extra_points) > 0:
        V_vol = np.vstack([V_surf, extra_points])
        surf_map = np.arange(len(V_surf), dtype=np.int32)
    else:
        V_vol = V_surf.copy()
        surf_map = np.arange(len(V_surf), dtype=np.int32)

    dela = Delaunay(V_vol)  # convex hull tets
    T = dela.simplices.astype(np.int32)
    return V_vol, T, surf_map, dela

def make_deformation_interpolator(V_vol, T, U_vol):
    """
    Build a callable u(x) that evaluates deformation at query points by
    locating the containing tet and barycentrically interpolating U.
    Falls back to nearest node if point is outside the tet mesh.

    Returns:
      ufunc(points: (k,3)) -> (k,3)
    """
    # Use Delaunay on volume vertices to locate simplices
    dela = Delaunay(V_vol)

    def eval_u(P):
        P = np.atleast_2d(P)
        simplex = dela.find_simplex(P)  # -1 for outside
        X = dela.transform  # shape (mt, 4, 3): affine maps for barycentric computation
        # For each point, compute barycentric coords if inside
        U = np.zeros((P.shape[0], 3), dtype=np.float64)
        inside = simplex >= 0
        idx = simplex[inside]
        # barycentric for Delaunay:
        # b[1:4] = T * (p - r), b0 = 1 - sum(b[1:4])
        Txf = X[idx, :3, :]           # (ki,3,3)
        r = X[idx, 3, :]              # (ki,3)
        q = (P[inside] - r) @ Txf.transpose(0,2,1)  # (ki,3)
        bary = np.column_stack([1.0 - q.sum(axis=1), q])  # (ki,4)

        # gather tet vertex indices
        tet_indices = T[idx]  # (ki,4)
        U[inside] = np.einsum('ki, kij -> kj', bary, U_vol[tet_indices])  # (ki,3)

        # For outside points, use nearest vertex displacement (simple, robust)
        if np.any(~inside):
            from scipy.spatial import cKDTree
            tree = cKDTree(V_vol)
            _, nn = tree.query(P[~inside], k=1)
            U[~inside] = U_vol[nn]
        return U

    return eval_u

# ---------------- Main pipeline ----------------

def volumetric_harmonic_warp_from_surfaces(V_src_surf, F_surf, V_tgt_surf,
                                           tetmesher='delaunay',
                                           extra_points=None,
                                           custom_tet=None):
    """
    Inputs:
      V_src_surf (n,3): source surface vertices
      F_surf (m,3):    source surface triangles (topology shared with target)
      V_tgt_surf (n,3): target surface vertices (same topology as source)
      tetmesher: 'delaunay' (default) or 'custom'
      extra_points: optional (k,3) interior Steiner points to improve Delaunay
      custom_tet: optional tuple (V_vol, T, surf_map) if you already have a tet mesh

    Returns:
      V_def_surf (n,3): deformed surface (≈ V_tgt_surf)
      U_vol (nv,3): volumetric displacement at tet vertices
      V_vol (nv,3), T (mt,4): volume mesh used
      ufunc: callable P(k,3)->(k,3) returning displacement at arbitrary 3D points
    """
    # 1) Tetrahedralize the source volume
    if custom_tet is not None:
        V_vol, T, surf_map = custom_tet
        dela_for_interp = Delaunay(V_vol)
    elif tetmesher == 'delaunay':
        V_vol, T, surf_map, dela_for_interp = simple_tetrahedralize_from_surface(
            V_src_surf, F_surf, extra_points=extra_points
        )
    else:
        raise ValueError("Unsupported 'tetmesher' option without custom_tet.")

    nv = V_vol.shape[0]
    n_surf = V_src_surf.shape[0]

    # 2) Set Dirichlet displacements on surface vertices (as boundary of the volume)
    # Here we assume the first n_surf vertices of V_vol correspond to surface vertices,
    # or surf_map explicitly maps them.
    bmask = np.zeros(nv, dtype=bool)
    bmask[surf_map] = True

    U_b = np.zeros((bmask.sum(), 3), dtype=np.float64)
    U_b[:] = (V_tgt_surf - V_src_surf)[np.arange(n_surf)]  # same topology

    # 3) Assemble volumetric Laplacian and solve for interior
    K = assemble_tet_laplacian(V_vol, T)
    interior = ~bmask
    if interior.sum() == 0:
        # fully constrained (uncommon) — just return target & zero field
        U_vol = np.zeros_like(V_vol)
        U_vol[bmask] = U_b
        V_def_surf = V_src_surf + U_b
        ufunc = make_deformation_interpolator(V_vol, T, U_vol)
        return V_def_surf, U_vol, V_vol, T, ufunc

    K_ii = K[interior][:, interior]
    K_ib = K[interior][:, bmask]
    rhs = -K_ib @ U_b

    U_vol = np.zeros((nv,3), dtype=np.float64)
    from scipy.sparse.linalg import factorized
    # Factorize once, solve 3 RHS
    solve_ii = factorized(K_ii.tocsc())
    for d in range(3):
        U_vol[interior, d] = solve_ii(rhs[:, d])
        U_vol[bmask, d] = U_b[:, d]

    # 4) Deformed surface = source surface + boundary displacement
    V_def_surf = V_src_surf + U_vol[surf_map]

    # 5) Deformation-field evaluator u(x)
    ufunc = make_deformation_interpolator(V_vol, T, U_vol)
    return V_def_surf, U_vol, V_vol, T, ufunc


# ---------------- Example ----------------
if __name__ == "__main__":
    # ---------------- Cube surface mesh ----------------
    # Vertices of unit cube [0,1]^3
    V_src = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # bottom square
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]  # top square
    ], dtype=np.float64)

    # Triangles (12 total, 2 per face)
    F = np.array([
        [0, 1, 2], [0, 2, 3],  # bottom
        [4, 5, 6], [4, 6, 7],  # top
        [0, 1, 5], [0, 5, 4],  # front
        [1, 2, 6], [1, 6, 5],  # right
        [2, 3, 7], [2, 7, 6],  # back
        [3, 0, 4], [3, 4, 7],  # left
    ], dtype=np.int32)

    # ---------------- Target cube deformation ----------------
    # For example: lift the top face up by +0.5 in z, and shear in x
    V_tgt = V_src.copy()
    top = V_src[:, 2] > 0.5  # vertices with z=1
    V_tgt[top, 2] += 0.5  # raise top
    V_tgt[top, 0] += 0.2  # shear right

    V_def_surf, U_vol, V_vol, T, u = volumetric_harmonic_warp_from_surfaces(
        V_src, F, V_tgt, tetmesher='delaunay'
    )
    print("Deformed surface vertices:\n", V_def_surf)
    # Query deformation at arbitrary points (e.g., embedded dense mesh vertices)
    pts = np.array([[0.5, 0.5, 0.02], [0.8, 0.9, 0.05]])
    # print("u(pts):\n", u(pts))
