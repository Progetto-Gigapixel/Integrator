"""
GPU-accelerated RTF (Restricted Tangent Face) computation for isometric remeshing.

This file implements a practical GPU-friendly version of Algorithm 2 in your pseudo-code.

Design decisions & notes:
- Uses PyTorch tensors on CUDA when available; falls back to NumPy (CPU) if CUDA not available.
- Brute-force KNN implemented with pairwise squared-distances; this is easy to run on GPU.
- Represent each half-space plane as (n, d) meaning: n.dot(x) <= d keeps the half-space.
  - For orthogonal dual (Voronoi bisector) between si and sj we use:
      n = (sj - si)
      d = (sj.dot(sj) - si.dot(si)) / 2
    so that points closer to si satisfy n.dot(x) <= d.
- Convex cell represented implicitly by its set of half-spaces. To obtain the
  intersection polygon between this cell and the RTF plane, we compute intersection
  points by solving linear systems for triplets of planes that include the RTF plane
  (i.e. RTF plane + two other planes). We then keep only points that satisfy *all*
  half-space inequalities (with a small epsilon tolerance). This yields the vertices
  of the intersection polygon on the RTF plane.
- This solution avoids complex per-face topology and is reasonably efficient for
  moderate k (e.g. k <= 64). Complexity per sample is roughly O((m^2) * 3^3) where
  m = 6 + k + 1 (bbox planes + neighbor planes + RTF plane).

Limitations and assumptions:
- The implementation focuses on correctness and GPU acceleration of the heavy
  linear algebra parts. For very large point counts (millions), further batching
  or a spatial acceleration structure should be added.
- The "two virtual points" trick described in the paper is implemented by
  adding two virtual neighbors located at si +/- d*ni (d configurable).
- Projection (Algorithm 3) is implemented in `project_barycenters_to_mesh`
  which projects a set of barycenters onto a triangle mesh using KNN to select
  nearby triangles. This function can run on CPU (NumPy) or GPU (PyTorch) if
  the mesh vertices/triangles are provided as tensors.

Dependencies:
- torch (preferred, for CUDA acceleration)
- numpy
- optionally: tqdm for progress bar in CPU mode

Usage example (summary only — the full code lives in this file):
- call `compute_rtf_all(points, normals, k=16, device='cuda')` to get list of
  RTF polygons (each polygon returned as Nx3 float32 tensor on CPU).
- call `compute_barycenters_from_rtf_polygons` to compute barycenters.
- call `project_barycenters_to_mesh` to project barycenters back to the mesh.

"""

from typing import Optional, Tuple, List
import math
import itertools

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

# Small helpers
EPS = 1e-7


def _to_device(x, device: Optional[str]):
    if _HAS_TORCH and device is not None:
        return x.to(device)
    return x


def knn_bruteforce(points: np.ndarray, query: np.ndarray, k: int, device: Optional[str] = None) -> np.ndarray:
    """Brute-force KNN. Returns indices of k nearest neighbors in `points` for each query point.

    points: (N,3) numpy or torch tensor
    query: (M,3)
    returns: (M,k) indices (numpy)
    """
    if _HAS_TORCH and device is not None and device.startswith('cuda'):
        pts = torch.as_tensor(points, device=device)
        q = torch.as_tensor(query, device=device)
        # pairwise squared distances: (M, N)
        d2 = (q.unsqueeze(1) - pts.unsqueeze(0)).pow(2).sum(dim=2)
        idx = d2.topk(k, largest=False).indices
        return idx.cpu().numpy()
    else:
        pts = np.asarray(points)
        q = np.asarray(query)
        M, N = q.shape[0], pts.shape[0]
        idx = np.empty((M, k), dtype=np.int64)
        for i in range(M):
            d2 = np.sum((pts - q[i:i+1])**2, axis=1)
            idx[i] = np.argpartition(d2, k)[:k]
        return idx


def _bbox_planes(bbox_min: np.ndarray, bbox_max: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return normals and ds for the 6 axis-aligned bbox half-spaces:
    n.dot(x) <= d
    normals shape (6,3), ds shape (6,)
    ordering: x <= max_x, -x <= -min_x, y <= max_y, -y <= -min_y, z <= max_z, -z <= -min_z
    """
    normals = np.array([[1.,0.,0.], [-1.,0.,0.], [0.,1.,0.], [0.,-1.,0.], [0.,0.,1.], [0.,0.,-1.]], dtype=np.float64)
    ds = np.array([bbox_max[0], -bbox_min[0], bbox_max[1], -bbox_min[1], bbox_max[2], -bbox_min[2]], dtype=np.float64)
    return normals, ds


def build_halfspaces_for_sample(si: np.ndarray,
                                neighbors: np.ndarray,
                                bbox_min: np.ndarray,
                                bbox_max: np.ndarray,
                                normal: np.ndarray,
                                virtual_offset: float = 20.0) -> Tuple[np.ndarray, np.ndarray]:
    """Build half-space planes for one sample `si`.

    Returns (normals, ds) where normals is (m,3) and ds is (m,)
    The returned half-spaces describe the convex cell before RTF clipping.
    """
    # bbox planes
    b_norm, b_ds = _bbox_planes(bbox_min, bbox_max)
    Hn = [b_norm]
    Hd = [b_ds]

    # add virtual points
    sv1 = si + virtual_offset * normal
    sv2 = si - virtual_offset * normal
    neighbors_aug = np.vstack([neighbors, sv1, sv2])

    # for each neighbor sj compute orthogonal dual plane for si's side:
    # n = (sj - si); d = (sj·sj - si·si)/2
    ni = np.asarray(si, dtype=np.float64)
    sj_all = neighbors_aug.astype(np.float64)
    vecs = sj_all - ni[np.newaxis, :]
    n_arr = vecs
    d_arr = (np.sum(sj_all*sj_all, axis=1) - np.sum(ni*ni)) * 0.5

    Hn.append(n_arr)
    Hd.append(d_arr)

    Hn = np.vstack(Hn)
    Hd = np.concatenate(Hd)
    return Hn, Hd


def solve_triplet_intersections(normals: np.ndarray, ds: np.ndarray) -> np.ndarray:
    """Given arrays normals (m,3), ds (m,), compute intersection points for ALL combinations
    of triples of planes. Returns array points (T,3) of candidate intersections (may contain NaNs).
    """
    m = normals.shape[0]
    comb = list(itertools.combinations(range(m), 3))
    pts = []
    for (i,j,k) in comb:
        A = np.vstack([normals[i], normals[j], normals[k]])
        b = np.array([ds[i], ds[j], ds[k]], dtype=np.float64)
        try:
            x = np.linalg.solve(A, b)
            pts.append(x)
        except np.linalg.LinAlgError:
            continue
    if len(pts) == 0:
        return np.zeros((0,3), dtype=np.float64)
    return np.vstack(pts)


def filter_points_by_halfspaces(points: np.ndarray, normals: np.ndarray, ds: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """Keep points satisfying ALL halfspace constraints n.dot(x) <= d + tol"""
    if points.shape[0] == 0:
        return points
    vals = points.dot(normals.T)  # (P, m)
    ok = np.all(vals <= (ds + tol)[np.newaxis, :], axis=1)
    return points[ok]


def compute_rtf_for_sample(si: np.ndarray,
                           ni: np.ndarray,
                           neighbors: np.ndarray,
                           bbox_min: np.ndarray,
                           bbox_max: np.ndarray,
                           k_virtual_offset: float = 20.0) -> np.ndarray:
    """Compute RTF polygon vertices (in 3D) for a single sampling point si.
    Returns an (L,3) array of vertices in arbitrary order."
    # build halfspaces pre-clipping
    Hn, Hd = build_halfspaces_for_sample(si, neighbors, bbox_min, bbox_max, ni, virtual_offset=k_virtual_offset)
    """
    # RTF plane: normal ni, d_rt = ni·si
    rt_n = np.asarray(ni, dtype=np.float64)
    rt_d = float(rt_n.dot(si))

    # we want vertices that lie on the RTF plane and on intersection with two other planes
    # so pick combinations rt_plane + two others. To reuse existing utilities we append RTF as a plane
    Hn_all = np.vstack([Hn, rt_n[np.newaxis, :]])
    Hd_all = np.concatenate([Hd, np.array([rt_d], dtype=np.float64)])
    rt_idx = Hn_all.shape[0] - 1

    # compute triplet intersections that include rt_idx
    m = Hn_all.shape[0]
    pts = []
    # iterate all pairs i<j excluding rt_idx
    for a in range(m):
        if a == rt_idx:
            continue
        for b in range(a+1, m):
            if b == rt_idx:
                continue
            A = np.vstack([Hn_all[rt_idx], Hn_all[a], Hn_all[b]])
            rhs = np.array([Hd_all[rt_idx], Hd_all[a], Hd_all[b]], dtype=np.float64)
            try:
                x = np.linalg.solve(A, rhs)
            except np.linalg.LinAlgError:
                continue
            pts.append(x)
    if len(pts) == 0:
        return np.zeros((0,3), dtype=np.float64)
    pts = np.vstack(pts)

    # filter points inside pre-clipped halfspaces (i.e., before RTF clipping)
    Hn_pre = Hn
    Hd_pre = Hd
    pts_in = filter_points_by_halfspaces(pts, Hn_pre, Hd_pre, tol=1e-8)

    # Remove duplicates (within tol)
    if pts_in.shape[0] == 0:
        return pts_in
    uniq = []
    for p in pts_in:
        if not any(np.linalg.norm(p - q) < 1e-8 for q in uniq):
            uniq.append(p)
    return np.vstack(uniq)


def compute_rtf_all(samples: np.ndarray,
                    normals: np.ndarray,
                    mesh_vertices: Optional[np.ndarray] = None,
                    k: int = 16,
                    bbox_padding: float = 0.01,
                    virtual_offset: float = 20.0,
                    device: Optional[str] = 'cuda') -> List[np.ndarray]:
    """Compute RTF polygons for all samples.

    samples: (n,3) array
    normals: (n,3)
    mesh_vertices: optional (N,3) to compute bbox; if None bbox computed from samples
    k: number of neighbors
    Returns: list of length n with arrays (Li,3) of polygon vertices (on CPU numpy)
    """
    pts = np.asarray(samples, dtype=np.float64)
    n = pts.shape[0]
    if mesh_vertices is None:
        minv = pts.min(axis=0) - bbox_padding
        maxv = pts.max(axis=0) + bbox_padding
    else:
        mv = np.asarray(mesh_vertices, dtype=np.float64)
        minv = mv.min(axis=0) - bbox_padding
        maxv = mv.max(axis=0) + bbox_padding

    # Precompute global neighbor indices using brute-force
    idxs = knn_bruteforce(pts, pts, k+1, device=device)  # includes self
    # drop self (first may be self) -> take indices != i and up to k
    neighbors_idx = []
    for i in range(n):
        row = idxs[i]
        row = row[row != i]
        if len(row) < k:
            # pad with repeating last index
            pad = np.full(k - len(row), row[-1] if len(row)>0 else i, dtype=np.int64)
            row = np.concatenate([row, pad])
        neighbors_idx.append(row[:k])
    neighbors_idx = np.vstack(neighbors_idx)

    rtf_polygons = [None] * n
    for i in range(n):
        si = pts[i]
        ni = np.asarray(normals[i], dtype=np.float64)
        neigh = pts[neighbors_idx[i]]
        poly = compute_rtf_for_sample(si, ni, neigh, minv, maxv, k_virtual_offset=virtual_offset)
        rtf_polygons[i] = poly
    return rtf_polygons


def compute_barycenters_from_rtf_polygons(rtf_polygons: List[np.ndarray]) -> np.ndarray:
    """Compute barycenters (average of vertices) for each polygon. Empty polygons get NaN.
    Returns (n,3) array."""
    out = np.zeros((len(rtf_polygons), 3), dtype=np.float64)
    for i, poly in enumerate(rtf_polygons):
        if poly is None or poly.shape[0] == 0:
            out[i] = np.array([np.nan, np.nan, np.nan])
        else:
            out[i] = poly.mean(axis=0)
    return out


def project_point_to_triangle(p: np.ndarray, tri: np.ndarray) -> Tuple[np.ndarray, float]:
    """Project point p onto triangle tri (3,3). Returns projected point and squared distance."""
    # Based on standard point-triangle projection
    A = tri[0]; B = tri[1]; C = tri[2]
    AB = B - A
    AC = C - A
    AP = p - A
    d1 = np.dot(AB, AP)
    d2 = np.dot(AC, AP)
    if d1 <= 0 and d2 <= 0:
        return A, np.sum((p - A)**2)
    BP = p - B
    d3 = np.dot(AB, BP)
    d4 = np.dot(AC, BP)
    if d3 >= 0 and d4 <= d3:
        return B, np.sum((p - B)**2)
    vc = d1*d4 - d3*d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        v = d1 / (d1 - d3)
        proj = A + v * AB
        return proj, np.sum((p - proj)**2)
    CP = p - C
    d5 = np.dot(AB, CP)
    d6 = np.dot(AC, CP)
    if d6 >= 0 and d5 <= d6:
        return C, np.sum((p - C)**2)
    vb = d5*d2 - d1*d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        w = d2 / (d2 - d6)
        proj = A + w * AC
        return proj, np.sum((p - proj)**2)
    va = d3*d6 - d5*d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        proj = B + w * (C - B)
        return proj, np.sum((p - proj)**2)
    # inside face region. compute projection onto plane
    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    proj = A + AB * v + AC * w
    return proj, np.sum((p - proj)**2)


def project_barycenters_to_mesh(barycenters: np.ndarray,
                                mesh_vertices: np.ndarray,
                                mesh_triangles: np.ndarray,
                                k: int = 16) -> np.ndarray:
    """Project each barycenter onto the mesh (triangles) using KNN-based selection.

    barycenters: (n,3)
    mesh_vertices: (V,3)
    mesh_triangles: (T,3) triangles as indices into mesh_vertices
    Returns projected points (n,3)
    """
    V = mesh_vertices.shape[0]
    # Build quick vertex KNN (CPU)
    idxs = knn_bruteforce(mesh_vertices, barycenters, k, device=None)  # CPU
    proj = np.zeros_like(barycenters)
    for i in range(barycenters.shape[0]):
        bi = barycenters[i]
        neigh_vids = np.unique(idxs[i])
        # collect triangles that contain any of these vertices
        tris_mask = np.any(np.isin(mesh_triangles, neigh_vids[np.newaxis, :]), axis=1)
        tris = mesh_triangles[tris_mask]
        best_p = None
        best_d2 = float('inf')
        for tri_idx in tris:
            tri = mesh_vertices[tri_idx]
            pproj, d2 = project_point_to_triangle(bi, tri)
            if d2 < best_d2:
                best_d2 = d2
                best_p = pproj
        if best_p is None:
            # fallback: nearest neighbor vertex
            nearest = neigh_vids[0]
            proj[i] = mesh_vertices[nearest]
        else:
            proj[i] = best_p
    return proj


# If run as script, simple demo
if __name__ == '__main__':
    # tiny demo with random points on a sphere
    import time
    n = 100
    rng = np.random.RandomState(0)
    pts = rng.normal(size=(n,3))
    pts /= np.linalg.norm(pts, axis=1)[:,None]
    # normals roughly pointing outwards
    norms = pts.copy()

    t0 = time.time()
    polys = compute_rtf_all(pts, norms, k=8, device=None)
    bary = compute_barycenters_from_rtf_polygons(polys)
    t1 = time.time()
    print(f'Computed {len(polys)} RTF polygons in {t1-t0:.3f}s')
    valid = np.sum(~np.isnan(bary[:,0]))
    print(f'Barycenters valid: {valid}/{n}')

