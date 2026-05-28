import numpy as np
import pytest

# importa le funzioni che vuoi testare
from gpu_rtf_remeshing import (
    _bbox_planes,
    knn_bruteforce,
    build_halfspaces_for_sample,
    compute_rtf_for_sample,
    compute_rtf_all,
    compute_barycenters_from_rtf_polygons,
    project_point_to_triangle,
    project_barycenters_to_mesh,
)


def almost_equal(a, b, tol=1e-6):
    return np.linalg.norm(a - b) < tol


def test_bbox_planes():
    # test semplice: bounding box da (0,0,0) a (1,1,1)
    normals, ds = _bbox_planes(np.array([0.,0.,0.]), np.array([1.,1.,1.]))
    # Le sei semispazi dovrebbero includere il punto (0.5,0.5,0.5)
    p = np.array([0.5,0.5,0.5])
    vals = np.dot(normals, p)
    assert np.all(vals <= ds + 1e-8)


def test_knn_bruteforce_small():
    # punti semplici su linea
    pts = np.array([[0.,0.,0.], [1.,0.,0.], [2.,0.,0.], [3.,0.,0.]])
    query = np.array([[1.1,0,0], [2.9,0,0]])
    # k=2
    idx = knn_bruteforce(pts, query, k=2, device=None)
    # per il primo, i due vicini più vicini a 1.1 sono punti a 1 e 0 o 2
    # controlliamo che il primo vicino per query[0] sia 1 o 2
    assert 1 in idx[0] or 2 in idx[0]
    # controlliamo che per query[1] uno dei vicini sia l'indice 3
    assert 3 in idx[1]


def test_build_halfspaces_virtual_points():
    si = np.array([0.,0.,0.])
    normal = np.array([0.,0.,1.])
    neighbor = np.array([[1.,0.,0.]])
    normals, ds = build_halfspaces_for_sample(si, neighbor, bbox_min=np.array([-1,-1,-1]), bbox_max=np.array([1,1,1]), normal=normal, virtual_offset=5.0)
    # Ci dovrebbero essere: 6 bbox piani + 1 neighbor ortogonale + 2 virtual neighbor piani = almeno 9 planes
    assert normals.shape[0] >= 9
    assert ds.shape[0] == normals.shape[0]


def test_compute_rtf_for_sample_plane_cut():
    # Punti su piano z=0, normale z. Dovremmo ottenere intersezione del bounding box + bisettrici
    si = np.array([0.,0.,0.])
    ni = np.array([0.,0.,1.])
    # vicini: un punto in alto e uno in basso
    neighbors = np.array([[0.,0.,2.], [0.,0.,-2.]])
    bbox_min = np.array([-1,-1,-1])
    bbox_max = np.array([1,1,1])
    poly = compute_rtf_for_sample(si, ni, neighbors, bbox_min, bbox_max, virtual_offset=1.0)
    # L'intersezione del piano z=0 con il box [-1,1]^3 dovrebbe essere un quadrato in z=0 su [-1,1]×[-1,1]
    # Quindi 4 vertici: (±1,±1,0)
    # Controlliamo che ci siano almeno questi 4 punti (ordine non importa)
    expected = set([
        (-1., -1., 0.),
        (-1.,  1., 0.),
        ( 1., -1., 0.),
        ( 1.,  1., 0.),
    ])
    # arrotondiamo i punti ottenuti
    rounded = set(tuple(np.round(p, 5)) for p in poly)
    assert expected.issubset(rounded)


def test_compute_rtf_all_and_barycenters():
    # un set di 4 punti sui vertici di un quadrato su xy-plane, normali verso z
    samples = np.array([
        [0.5, 0.5, 0.],
        [0.5, -0.5, 0.],
        [-0.5, 0.5, 0.],
        [-0.5, -0.5, 0.],
    ])
    normals = np.tile(np.array([0.,0.,1.]), (4,1))
    polys = compute_rtf_all(samples, normals, k=2, device=None)
    bary = compute_barycenters_from_rtf_polygons(polys)
    # Tutti i baricentri dovrebbero avere z ≈ 0
    for b in bary:
        assert pytest.approx(0.0, abs=1e-6) == b[2]
    # I barycentri dovrebbero rimanere dentro il bounding box (≈ [-1,1]² in x,y)
    for b in bary:
        assert -2.0 < b[0] < 2.0
        assert -2.0 < b[1] < 2.0


def test_project_point_to_triangle_inside_and_edge():
    # un triangolo semplice nel piano xy
    tri = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    # punto interno
    p1 = np.array([0.2,0.2,0.1])
    proj1, d2_1 = project_point_to_triangle(p1, tri)
    # proiezione dovrebbe essere (0.2,0.2,0.0)
    assert almost_equal(proj1, np.array([0.2,0.2,0.0]))
    # distanza dovrebbe essere (0.1)^2 = 0.01
    assert pytest.approx(0.01, abs=1e-8) == d2_1

    # punto vicino al bordo (vicino edge AB)
    p2 = np.array([0.7,0.1,0.3])
    proj2, d2_2 = project_point_to_triangle(p2, tri)
    # Proiezione z=0
    assert pytest.approx(0.3**2, abs=1e-6) == d2_2
    assert abs(proj2[2]) < 1e-8


def test_project_barycenters_to_mesh_basic():
    # faccia semplice: un quadrato diviso in 2 triangoli
    mesh_vertices = np.array([
        [0.,0.,0.],
        [1.,0.,0.],
        [1.,1.,0.],
        [0.,1.,0.],
    ])
    mesh_triangles = np.array([
        [0,1,2],
        [0,2,3],
    ])
    bary = np.array([
        [0.5, 0.5, 0.2],  # sopra il centro
        [0.1, 0.9, -0.3], # sotto il bordo superiore
    ])
    proj = project_barycenters_to_mesh(bary, mesh_vertices, mesh_triangles, k=3)
    # dovrebbero essere proiettati sul piano z=0
    for p in proj:
        assert abs(p[2]) < 1e-8
    # il punto centrale dovrebbe proiettarsi al centro approx
    assert almost_equal(proj[0], np.array([0.5,0.5,0.0]))


if __name__ == '__main__':  # facilita esecuzione manuale
    pytest.main()
