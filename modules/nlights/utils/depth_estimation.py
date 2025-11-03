"""
Module for estimating depth from normal maps and gradients.
Equivalent to DepthFromGradient.m in MATLAB.
"""

import numpy as np
from scipy.fftpack import dct, idct
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import cv2 as cv
from scipy.sparse import diags, kron, eye, csc_matrix, csr_matrix
from scipy.sparse.linalg import cg as scipy_cg, spsolve, spilu, LinearOperator
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import numpy as np
from scipy.ndimage import gaussian_filter
# import torch
from tqdm import tqdm
import cupy as cp
from cupyx.scipy.sparse import csc_matrix, eye, diags, kron
from cupyx.scipy.sparse.linalg import cg as cupy_cg
import time




# def depth_from_gradient_t2(p, q, options=None):
#     """
#     Estimate depth map Z from gradient fields p and q such that:
#         dZ/dx = p, dZ/dy = q
#
#     Parameters:
#         p: numpy array (MxN), gradient along x
#         q: numpy array (MxN), gradient along y
#         options: dict, with key 'periodic' = True/False
#
#     Returns:
#         Z: reconstructed depth map
#     """
#     if options is None:
#         options = {}
#
#     # Parse options
#     periodic = options.get('periodic', False)
#
#     # Verifica le dimensioni degli input
#     assert p.shape == q.shape, "p and q must have the same shape"
#     M, N = p.shape
#
#     # Esegui il copy-flip se non periodico
#
#     if not periodic:
#         p = np.block([
#             [p, -np.fliplr(p)],
#             [np.flipud(p), -np.flipud(np.fliplr(p))]
#         ])
#         q = np.block([
#             [q, np.fliplr(q)],
#             [-np.flipud(q), -np.flipud(np.fliplr(q))]
#         ])
#         M *= 2
#         N *= 2
#
#     # Frequency indices
#     halfM = (M - 1) / 2
#     halfN = (N - 1) / 2
#     u, v = np.meshgrid(
#         np.arange(-np.ceil(halfN), np.floor(halfN) + 1),
#         np.arange(-np.ceil(halfM), np.floor(halfM) + 1)
#     )
#     u = np.fft.ifftshift(u)
#     v = np.fft.ifftshift(v)
#
#     # Fourier transforms of p and q
#     Fp = np.fft(p)
#     Fq = np.fft(q)
#
#     # Avoid division by zero
#     denom = (u / N) ** 2 + (v / M) ** 2
#     denom[denom == 0] = 1  # will zero out DC later
#
#     # Compute Fz
#     Fz = -1j / (2 * np.pi) * (u * Fp / N + v * Fq / M) / denom
#
#     # Set DC component
#     Fz[0, 0] = 0
#
#     # Inverse FFT to get Z
#     Z = np.real(np.fft2(Fz))
#
#     if not periodic:
#         Z = Z[:M//2, :N//2]
#
#     return Z



# def depth_from_normals___(normals, mask=None, options=None):
#     """
#     Estimate depth from surface normal vectors.
#
#     Parameters
#     ----------
#     normals : ndarray
#         Normal map (height, width, 3)
#     mask : ndarray, optional
#         Binary mask indicating valid pixels
#     options : dict, optional
#         Dictionary with options for depth_from_gradient
#
#     Returns
#     -------
#     ndarray
#         Estimated depth map
#     """
#     if options is None:
#         options = {}
#
#     # Extract height and width
#     height, width = normals.shape[:2]
#
#     # Extract gradient information from normals
#     # p = dz/dx, q = dz/dy
#     p = -normals[:, :, 0] / np.maximum(normals[:, :, 2], 1e-10)
#     q = -normals[:, :, 1] / np.maximum(normals[:, :, 2], 1e-10)
#
#     # Apply mask if provided
#     if mask is not None:
#         p = p * mask
#         q = q * mask
#
#     # Handle NaN and Inf values
#     p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
#     q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
#
#     # Compute depth using depth_from_gradient
#     Z = depth_from_gradient_t2(p, q, options)
#
#     # Apply mask to result if provided
#     if mask is not None:
#         Z = Z * mask
#
#     return Z

def depth_from_normals_matlab(normals):
    """
    Versione ottimizzata per replicare la versione di Yiong Xiang in pythonese
    """

    p = -normals[:, :, 0] / np.maximum(normals[:, :, 2], 1e-10)
    q = -normals[:, :, 1] / np.maximum(normals[:, :, 2], 1e-10)

    periodic = True
    M, N = p.shape
    p = p.astype(np.float64)
    q = q.astype(np.float64)

    if not periodic:
        p = np.block([
            [p, -p[:, ::-1]],
            [p[::-1, :], -p[::-1, ::-1]]
        ])
        q = np.block([
            [q, q[:, ::-1]],
            [-q[::-1, :], -q[::-1, ::-1]]
        ])
        M *= 2
        N *= 2

    u = np.fft.fftfreq(N)
    v = np.fft.fftfreq(M)
    u, v = np.meshgrid(u, v, indexing='xy')

    Fp = np.fft.fft2(p)
    Fq = np.fft.fft2(q)

    denom = (u ** 2 + v ** 2)
    denom[denom == 0] = np.finfo(float).eps

    Fz = -1j / (2 * np.pi) * (u * Fp + v * Fq) / denom
    Fz[0, 0] = 0  # opzionale: azzera la media se desiderato

    Z = np.real(np.fft.ifft2(Fz))

    if not periodic:
        Z = Z[:M//2, :N//2]

    # Non forziamo offset se non serve, oppure:
    # Z -= np.mean(Z)
    Z -= Z.min()

    return Z.astype(np.float32)

def depth_from_normals(normals):
    # Ricava i gradienti p = dZ/dx e q = dZ/dy dalle normali
    # normals = scipy.ndimage.gaussian_filter(normals, sigma=1.0)
    # Set depth to 0 for invalid normal vectors
    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    normals_normalized = normals / (norm + 1e-8)

    mask = np.logical_or.reduce((np.isnan(normals_normalized[:, :, 0]),
                                 np.isnan(normals_normalized[:, :, 1]),
                                 np.isnan(normals_normalized[:, :, 2])))

    nx, ny, nz = normals_normalized[..., 0], normals_normalized[..., 1], normals_normalized[..., 2]
    epsilon = 1e-6
    p = -nx / (nz + epsilon)
    q = -ny / (nz + epsilon)
    Z = - depth_from_gradient(p, q)
    Z[mask] = 0
    # Z = gaussian_filter(Z, sigma=10)
    return Z

def depth_from_gradient(p, q):
    M, N = p.shape
    u, v = np.meshgrid(np.fft.fftfreq(N), np.fft.fftfreq(M))
    u = np.fft.fftshift(u)
    v = np.fft.fftshift(v)

    Fp = np.fft.fft2(p)
    Fq = np.fft.fft2(q)

    denom = (u**2 + v**2)
    denom[denom == 0] = 1e-6

    Fz = -1j * (u * Fp + v * Fq) / (2 * np.pi * denom)
    Fz[0, 0] = 0  # azzera la componente DC
    Z = np.real(np.fft.ifft2(Fz))

    # Level Z to the lowest value
    offset = np.min(Z)
    Z = Z - offset

    return Z


def depth_from_normals_fft_filtered(normals, filter_strength=0.1):
    # Ricava i gradienti p = dZ/dx e q = dZ/dy dalle normali
    # normals = scipy.ndimage.gaussian_filter(normals, sigma=1.0)
    # Set depth to 0 for invalid normal vectors
    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    normals_normalized = normals / (norm + 1e-8)

    mask = np.logical_or.reduce((np.isnan(normals_normalized[:, :, 0]),
                                 np.isnan(normals_normalized[:, :, 1]),
                                 np.isnan(normals_normalized[:, :, 2])))

    nx, ny, nz = normals_normalized[..., 0], normals_normalized[..., 1], normals_normalized[..., 2]
    epsilon = 1e-6
    p = -nx / (nz + epsilon)
    q = -ny / (nz + epsilon)
    Z = - depth_from_gradient_filtered(p, q, filter_strength=0.1)
    Z[mask] = 0
    return Z


def depth_from_gradient_filtered(p, q, filter_strength=0.1):
    M, N = p.shape

    # Frequenze u,v
    u, v = np.meshgrid(np.fft.fftfreq(N), np.fft.fftfreq(M))
    u = np.fft.fftshift(u)
    v = np.fft.fftshift(v)

    # FFT dei gradienti
    Fp = np.fft.fft2(p)
    Fq = np.fft.fft2(q)

    # Filtro gaussiano passa-basso
    sigma = filter_strength
    gaussian = np.exp(- (u**2 + v**2) / (2 * sigma**2))
    gaussian = np.fft.ifftshift(gaussian)

    Fp_filtered = Fp * gaussian
    Fq_filtered = Fq * gaussian

    # Evita divisione per zero
    denom = (u**2 + v**2)
    denom[denom == 0] = 1e-6

    # Integrazione in frequenza
    Fz = -1j * (u * Fp_filtered + v * Fq_filtered) / (2 * np.pi * denom)
    Fz[0, 0] = 0  # componente DC

    Z = np.real(np.fft.ifft2(Fz))

    # Livella la profondità
    Z = Z - np.min(Z)

    return Z


def compute_depth_map(normals):
    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    normals = normals / (norm + 1e-8)

    h = normals.shape[0]
    w = normals.shape[1]
    P = np.zeros((h, w, 2), dtype=np.float32)
    Q = np.zeros((h, w, 2), dtype=np.float32)
    tempZ = np.zeros((h, w, 2), dtype=np.float32)
    Z = np.zeros((h, w), dtype=np.float32)
    landa = 1.0
    mu = 1.0
    # cv.dft(pgrads, P, cv.DFT_COMPLEX_OUTPUT)
    # cv.dft(qgrads, Q, cv.DFT_COMPLEX_OUTPUT)

    for i in range (1, h):
        for j in range (1, w):
            u = np.sin(i * 2 * np.pi / h)
            v = np.sin(j * 2 * np.pi / w)
            uv = np.float_power(u, 2) + np.float_power(v, 2)
            d = (1 + landa)*uv + mu*np.float_power(uv, 2)
            tempZ [i, j, 0] = (u*P[i, j, 1] + v*Q[i, j, 1]) / d
            tempZ [i, j, 1] = (-u*P[i, j, 0] - v*Q[i, j, 0]) / d
    tempZ[0, 0, 0] = 0
    tempZ[0, 0, 1] = 0
    flags = cv.DFT_INVERSE + cv.DFT_SCALE + cv.DFT_REAL_OUTPUT
    cv.dft(tempZ, Z, flags)
    z_norm = cv.normalize(Z, None, 0, 1, cv.NORM_MINMAX, cv.CV_8U)
    return z_norm

# def depth_from_gradient_poisson(normals, tolerance=1e-5, max_iteration=1000):
#     norm = np.linalg.norm(normals, axis=2, keepdims=True)
#     normals_normalized = normals / (norm + 1e-8)
#
#     mask = np.logical_or.reduce((np.isnan(normals_normalized[:, :, 0]),
#                                  np.isnan(normals_normalized[:, :, 1]),
#                                  np.isnan(normals_normalized[:, :, 2])))
#
#     nx, ny, nz = normals_normalized[..., 0], normals_normalized[..., 1], normals_normalized[..., 2]
#     epsilon = 1e-6
#     p = -nx / (nz + epsilon)
#     q = -ny / (nz + epsilon)
#
#     M, N = p.shape
#     dx = np.zeros_like(p)
#     dy = np.zeros_like(q)
#
#     dx[:, :-1] = p[:, :-1]
#     dx[:, -1] = 0
#     dy[:-1, :] = q[:-1, :]
#     dy[-1, :] = 0
#
#     # Divergenza (∇·(p, q))
#     f = dx - np.roll(dx, 1, axis=1) + dy - np.roll(dy, 1, axis=0)
#     f[0, :] = 0
#     f[:, 0] = 0
#
#     # Laplaciano discreto con condizioni di Dirichlet
#     e = np.ones(N)
#     Dxx = diags([e, -2*e, e], [-1, 0, 1], shape=(N, N))
#     Dyy = diags([e, -2*e, e], [-1, 0, 1], shape=(M, M))
#     L = kron(eye(M), Dxx) + kron(Dyy, eye(N))
#
#     rhs = f.flatten()
#     L = csc_matrix(L)
#
#     # Risoluzione sistema lineare (Poisson)
#     z_flat, info = cg(L, rhs, tol=tolerance, maxiter=max_iteration)
#     if info != 0:
#         print(f"Warning: cg did not converge, info = {info}")
#
#     Z = z_flat.reshape(M, N)
#     Z -= np.min(Z)
#     return Z


def depth_from_normals_misto_matlab(normals, periodic=False):
    """
    Ricostruzione della mappa di profondità da p = -dZ/dx e q = -dZ/dy
    con gestione dei bordi come in MATLAB.

    Parameters
    ----------
    periodic : bool
        Se True assume condizioni periodiche, altrimenti simmetrico.

    Returns
    -------
    Z : ndarray
        Mappa di profondità ricostruita
    """

    p = -normals[:, :, 0] / np.maximum(normals[:, :, 2], 1e-10)
    q = -normals[:, :, 1] / np.maximum(normals[:, :, 2], 1e-10)

    assert p.shape == q.shape
    M, N = p.shape

    if not periodic:
        # Estensione simmetrica
        p = np.block([
            [p, -p[:, ::-1]],
            [p[::-1, :], -p[::-1, ::-1]]
        ])
        q = np.block([
            [q, q[:, ::-1]],
            [-q[::-1, :], -q[::-1, ::-1]]
        ])
        M *= 2
        N *= 2

    # Indici delle frequenze (in stile MATLAB)
    halfM = (M - 1) / 2
    halfN = (N - 1) / 2
    u, v = np.meshgrid(
        np.arange(-np.ceil(halfN), np.floor(halfN) + 1),
        np.arange(-np.ceil(halfM), np.floor(halfM) + 1)
    )
    u = np.fft.ifftshift(u)
    v = np.fft.ifftshift(v)

    # FFT di p e q
    Fp = np.fft.fft2(p)
    Fq = np.fft.fft2(q)

    # Costruzione di Fz
    denom = (u / N) ** 2 + (v / M) ** 2
    denom[denom == 0] = 1e-6  # evitare divisione per zero

    Fz = -1j / (2 * np.pi) * ((u * Fp / N) + (v * Fq / M)) / denom
    Fz[0, 0] = 0  # Azzera componente DC

    # Inversa per ottenere Z
    Z = np.real(np.fft.ifft2(Fz))

    # Se non periodico, riprendi la porzione originale
    if not periodic:
        Z = Z[:M // 2, :N // 2]

    # Livella Z a partire da zero
    Z -= Z.min()

    return Z



def dct2(x):
    return dct(dct(x.T, norm='ortho').T, norm='ortho')

def idct2(x):
    return idct(idct(x.T, norm='ortho').T, norm='ortho')


def depth_from_normals2(normals):

    M, N, _ = normals.shape
    nx = normals[..., 0]
    ny = normals[..., 1]
    nz = normals[..., 2]
    epsilon = 1e-8

    # Calcola p e q dai normali
    p = -nx / (nz + epsilon)
    q = -ny / (nz + epsilon)

    # Calcola il termine f = div(p,q)
    dx = np.zeros_like(p)
    dy = np.zeros_like(q)

    dx[:, :-1] = p[:, :-1]
    dx[:, -1] = 0
    dy[:-1, :] = q[:-1, :]
    dy[-1, :] = 0

    f = dx - np.roll(dx, 1, axis=1) + dy - np.roll(dy, 1, axis=0)

    e = np.ones(N)
    D = diags([e, -2*e, e], offsets=[-1,0,1], shape=(N,N))
    I = eye(M)
    L = kron(I, D) + kron(D, I)  # Laplaciano 2D con condizioni di Neumann

    # Flatten del termine noto
    rhs = f.flatten()

    # Risolvo sistema lineare L * z = rhs
    z = spsolve(L, rhs)

    # Ricostruisco la depth map 2D
    Z = z.reshape(M, N)

    # Normalizzo per rendere minima depth a zero
    Z -= Z.min()

    return Z

def depth_from_normals_fft(normals):
    M, N, _ = normals.shape
    nx = normals[..., 0]
    ny = normals[..., 1]
    nz = normals[..., 2]
    epsilon = 1e-8

    # Calcola p e q dai normali
    p = -nx / (nz + epsilon)
    q = -ny / (nz + epsilon)

    fx = np.zeros_like(p)
    fy = np.zeros_like(q)

    fx[:, :-1] = p[:, :-1]
    fy[:-1, :] = q[:-1, :]

    f = fx - np.roll(fx, 1, axis=1) + fy - np.roll(fy, 1, axis=0)

    f_dct = dctn(f, norm='ortho')
    xx, yy = np.meshgrid(np.arange(N), np.arange(M))
    denom = 2 * (np.cos(np.pi * xx / N) - 1) + 2 * (np.cos(np.pi * yy / M) - 1)
    denom[0, 0] = 1  # per evitare divisione per zero

    Z_dct = f_dct / denom
    Z_dct[0, 0] = 0

    Z = idctn(Z_dct, norm='ortho')
    Z -= np.min(Z)
    return Z


def depth_from_gradient_poisson(normals):

    nx, ny, nz = normals[..., 0], normals[..., 1], normals[..., 2]
    epsilon = 1e-6
    p = -nx / (nz + epsilon)
    q = -ny / (nz + epsilon)

    M, N = p.shape
    print("Poisson: Computing divergencies from gradient field", flush=True)
    fx = np.zeros_like(p)
    fy = np.zeros_like(q)

    fx[:, :-1] = p[:, :-1]
    fy[:-1, :] = q[:-1, :]

    div = (fx - np.roll(fx, 1, axis=1)) + (fy - np.roll(fy, 1, axis=0))

    Ix = eye(N)
    Iy = eye(M)

    print("Poisson: Setting solver", flush=True)
    e = np.ones(N)
    Dx = diags([e, -2 * e, e], [-1, 0, 1], shape=(N, N))
    Dx = Dx.tolil()
    Dx[0, 0] = -1
    Dx[-1, -1] = -1

    e = np.ones(M)
    Dy = diags([e, -2 * e, e], [-1, 0, 1], shape=(M, M))
    Dy = Dy.tolil()
    Dy[0, 0] = -1
    Dy[-1, -1] = -1

    A = kron(Iy, Dx) + kron(Dy, Ix)

    print("Poisson: Solving the system", flush=True)
    b = div.flatten()

    ####
    # print("Poisson: Computing M", flush=True)
    # M_inv = diags(1 / A.diagonal())
    # M = LinearOperator(A.shape, matvec=lambda x: M_inv @ x)
    # print("Poisson: Computing M: Done", flush=True)
    ####
    t0 = time.time()
    Z_flat, info = cg(A, b, atol=1e-5, rtol=1e-5, maxiter=5000) #M=M
    elapsed = time.time() - t0
    print(f"Poisson: Solving time: {elapsed:.2f} seconds")

    if info != 0:
        print(f"Warning: solver did not converge (info={info})")

    Z = Z_flat.reshape((M, N))

    # Normalizza Z
    Z -= np.min(Z)

    print("Poisson: System solved", flush=True)
    return Z

## Parallelizzazione su GPU con CuPy e tiling
def depth_from_gradient_poisson_cupy_tiled(
    normals,
    tile_size=4096,
    overlap=256,
    max_gpu_memory_gb=8,
    n_schwarz_iters=1,
    relaxation=1.0,
    cg_tol=1e-5,
    cg_maxit=5000,
):
    """
    Versione CuPy che usa ESATTAMENTE la stessa matematica della versione CPU:
    - Stessa discretizzazione del gradiente -> divergenza
    - Stessi operatori 1D con prodotto di Kronecker
    - Stessa gestione dei bordi globali vs interni
    """
    
    # Estrai dimensioni
    M, N = normals.shape[:2]
    print(f"=== CuPy Tiled Poisson (CPU Math Compatible) ===")
    print(f"Input size: {M}×{N} = {M*N:,} px")

    # Stima memoria e ridimensionamento tile (stesso della versione originale)
    tile_points = tile_size * tile_size
    mem_per_tile_gb = (tile_points * 8 * 6) / (1024**3)
    if mem_per_tile_gb > max_gpu_memory_gb * 0.8:
        new_tile = int(tile_size * np.sqrt(max_gpu_memory_gb * 0.8 / mem_per_tile_gb))
        new_tile = max(new_tile, overlap + 64)
        print(f"[mem] riduco tile {tile_size}→{new_tile}")
        tile_size = new_tile
    # === UNA SOLA TILE -> RICADE NEL CASO NO TILING ===
    if tile_size >= min(M, N):
        print(f"=== Small Problem -> No Tiling ===")
        return depth_from_gradient_poisson_cupy(normals)

    # === STESSO CALCOLO GRADIENTE -> DIVERGENZA DELLA VERSIONE CPU ===
    nx, ny, nz = normals[..., 0], normals[..., 1], normals[..., 2]
    epsilon = 1e-6
    p = -nx / (nz + epsilon)
    q = -ny / (nz + epsilon)

    # Stessa discretizzazione della CPU version
    fx = np.zeros_like(p, dtype=np.float64)
    fy = np.zeros_like(q, dtype=np.float64)
    fx[:, :-1] = p[:, :-1]
    fy[:-1, :] = q[:-1, :]
    div = (fx - np.roll(fx, 1, axis=1)) + (fy - np.roll(fy, 1, axis=0))
    div = div.astype(np.float64, copy=False)

    # Soluzione globale iniziale
    Z = np.zeros((M, N), dtype=np.float64)

    # === STESSI OPERATORI 1D DELLA VERSIONE CPU ===
    def one_dim_op_cupy(n, is_global_left, is_global_right):
        """
        Stesso operatore 1D della versione CPU, ma costruito con CuPy sparse
        """
        # Costruzione manuale della matrice tridiagonale
        rows = []
        cols = []
        data = []
        
        for i in range(n):
            # Diagonale principale
            rows.append(i)
            cols.append(i)
            if (i == 0 and is_global_left) or (i == n-1 and is_global_right):
                data.append(-1.0)  # Bordi globali
            else:
                data.append(-2.0)  # Interno
            
            # Sovradiagonale
            if i < n - 1:
                rows.append(i)
                cols.append(i + 1)
                data.append(1.0)
            
            # Sottodiagonale  
            if i > 0:
                rows.append(i)
                cols.append(i - 1)
                data.append(1.0)
        
        # Converti a CuPy arrays
        rows = cp.asarray(rows, dtype=cp.int32)
        cols = cp.asarray(cols, dtype=cp.int32)
        data = cp.asarray(data, dtype=cp.float64)
        
        return csc_matrix((data, (rows, cols)), shape=(n, n), dtype=cp.float64)

    # Parametri tiling identici alla CPU version
    core = max(1, tile_size - 2 * overlap)
    step = core
    tiles_y = (M - overlap + step - 1) // step
    tiles_x = (N - overlap + step - 1) // step

    def tile_bounds(ty, tx):
        """Stessa funzione tile_bounds della CPU version"""
        y0_core = ty * step
        x0_core = tx * step
        y0 = max(0, y0_core - overlap)
        x0 = max(0, x0_core - overlap)
        y1 = min(M, y0_core + core + overlap)
        x1 = min(N, x0_core + core + overlap)
        cy0 = y0_core - y0
        cx0 = x0_core - x0
        cy1 = cy0 + min(core, y1 - y0 - cy0)
        cx1 = cx0 + min(core, x1 - x0 - cx0)
        return (y0, y1, x0, x1, cy0, cy1, cx0, cx1)

    print(f"Tiles: {tiles_y} x {tiles_x}  (core={core}, overlap={overlap})")

    def process_tile_cupy(ty, tx, div, Z):
        """
        Processo tile identico alla CPU version, ma eseguito su GPU
        """
        y0, y1, x0, x1, cy0, cy1, cx0, cx1 = tile_bounds(ty, tx)
        if cy1 <= cy0 or cx1 <= cx0:
            return None

        # Finestre (su CPU)
        f_win = div[y0:y1, x0:x1]
        z_win = Z[y0:y1, x0:x1]
        
        # Core da risolvere
        mh = cy1 - cy0
        nh = cx1 - cx0
        f_core = f_win[cy0:cy1, cx0:cx1].copy()
        
        # === STESSA LOGICA DI INTERFACCIA DELLA CPU VERSION ===
        has_top = (cy0 > 0)
        has_bottom = (cy1 < (y1 - y0))
        has_left = (cx0 > 0)
        has_right = (cx1 < (x1 - x0))
        
        if has_top:
            b_top = z_win[cy0 - 1, cx0:cx1]
            f_core[0, :] += b_top
        if has_bottom:
            b_bottom = z_win[cy1, cx0:cx1]
            f_core[-1, :] += b_bottom
        if has_left:
            b_left = z_win[cy0:cy1, cx0 - 1]
            f_core[:, 0] += b_left
        if has_right:
            b_right = z_win[cy0:cy1, cx1]
            f_core[:, -1] += b_right
        
        # === STESSI FLAGS PER BORDI GLOBALI ===
        global_top = (y0 + cy0 == 0)
        global_bottom = (y0 + cy1 == M)
        global_left = (x0 + cx0 == 0)
        global_right = (x0 + cx1 == N)
        
        # === STESSI OPERATORI 1D CON KRONECKER ===
        # Trasferimento su GPU
        f_core_cp = cp.asarray(f_core, dtype=cp.float64)
        
        # Costruzione operatori identica alla CPU version
        Tx = one_dim_op_cupy(nh, is_global_left=global_left, is_global_right=global_right)
        Ty = one_dim_op_cupy(mh, is_global_left=global_top, is_global_right=global_bottom)
        Ix = eye(nh, format='csc', dtype=cp.float64)
        Iy = eye(mh, format='csc', dtype=cp.float64)
        
        A = kron(Iy, Tx) + kron(Ty, Ix)  # Stesso segno/schema del globale
        rhs = f_core_cp.ravel()
        
        # CG con stessi parametri della CPU version
        z_vec, info = cupy_cg(A, rhs, tol=cg_tol, maxiter=cg_maxit)
        if info != 0:
            pass  # Stessa gestione della CPU version
            
        z_core = z_vec.reshape((mh, nh))
        
        # Ritorna risultato su CPU
        z_core_cpu = cp.asnumpy(z_core)
        return (y0 + cy0, y0 + cy1, x0 + cx0, x0 + cx1, z_core_cpu)

    # === LOOP ITERATIVO SU TILES ===
    t_all = time.time()

    try:
        for it in range(1, n_schwarz_iters + 1):
            for ty in range(tiles_y):
                for tx in range(tiles_x):
                    t_it = time.time()
                    result = process_tile_cupy(ty, tx, div, Z)
                    if result is not None:
                        y_start, y_end, x_start, x_end, z_core = result
                        # Applicazione rilassamento (opzionale, non presente nella CPU version)
                        if relaxation != 1.0:
                            z_old = Z[y_start:y_end, x_start:x_end]
                            Z[y_start:y_end, x_start:x_end] = (1.0 - relaxation) * z_old + relaxation * z_core
                        else:
                            Z[y_start:y_end, x_start:x_end] = z_core
                    t_end = time.time()
                    print(f"[Iter {it}/{n_schwarz_iters}] tile ({ty}, {tx}) processed in {t_end - t_it:.2f}s")
            # Libera memoria GPU
            cp.get_default_memory_pool().free_all_blocks()
    except Exception as e:
        print(f"[Iter {it}/{n_schwarz_iters}] error:")
        print(e)
        raise e
    print(f"[Iter {it}/{n_schwarz_iters}] sweep done in {time.time()-t_all:.2f}s")

    # === STESSA NORMALIZZAZIONE DELLA CPU VERSION ===
    Z -= np.min(Z)
    print(f"=== CuPy tiled solve complete in {time.time()-t_all:.2f}s ===")
    return Z

def depth_from_gradient_poisson_cupy( 
    normals,
    cg_tol=1e-5,
    cg_maxit=5000,):
    # Operatori 1D come nella CPU (Neumann modificata sui bordi globali)
    def one_dim_op_cupy(n, is_global_left, is_global_right):
        rows, cols, data = [], [], []
        for i in range(n):
            rows.append(i); cols.append(i)
            data.append(-1.0 if (i == 0 and is_global_left) or (i == n-1 and is_global_right) else -2.0)
            if i < n-1:
                rows.append(i); cols.append(i+1); data.append(1.0)
            if i > 0:
                rows.append(i); cols.append(i-1); data.append(1.0)
        rows = cp.asarray(rows, dtype=cp.int32)
        cols = cp.asarray(cols, dtype=cp.int32)
        data = cp.asarray(data, dtype=cp.float64)
        return csc_matrix((data, (rows, cols)), shape=(n, n), dtype=cp.float64)
    t_all = time.time()
    M, N = normals.shape[:2]
    nx, ny, nz = normals[..., 0], normals[..., 1], normals[..., 2]
    epsilon = 1e-6
    p = -nx / (nz + epsilon)
    q = -ny / (nz + epsilon)

    # Stessa discretizzazione della CPU version
    fx = np.zeros_like(p, dtype=np.float64)
    fy = np.zeros_like(q, dtype=np.float64)
    fx[:, :-1] = p[:, :-1]
    fy[:-1, :] = q[:-1, :]
    div = (fx - np.roll(fx, 1, axis=1)) + (fy - np.roll(fy, 1, axis=0))
    div = div.astype(np.float64, copy=False)

    # Soluzione globale iniziale
    Z = np.zeros((M, N), dtype=np.float64)

    # Costruisci A = kron(Iy, Tx) + kron(Ty, Ix)
    Tx = one_dim_op_cupy(N, is_global_left=True,  is_global_right=True)
    Ty = one_dim_op_cupy(M, is_global_left=True,  is_global_right=True)
    Ix = eye(N, format='csc', dtype=cp.float64)
    Iy = eye(M, format='csc', dtype=cp.float64)
    A  = kron(Iy, Tx) + kron(Ty, Ix)

    rhs = cp.asarray(div, dtype=cp.float64).ravel()
    z_vec, info = cupy_cg(A, rhs, tol=cg_tol, maxiter=cg_maxit)
    if info != 0:
        print(f"[single-tile] CG ha restituito info={info} (continua comunque)")

    print(f"=== CuPy Poisson solve complete in {time.time()-t_all:.2f}s ===")
    Z = cp.asnumpy(z_vec.reshape(M, N))
    Z -= Z.min()
    return Z


def depth_from_gradient_torch(normals, device='cuda'):

    torch_device = torch.device(device)
    normals = torch.tensor(normals, device=torch_device, dtype=torch.float32)

    nx, ny, nz = normals[..., 0], normals[..., 1], normals[..., 2]
    epsilon = 1e-6
    p = -nx / (nz + epsilon)
    q = -ny / (nz + epsilon)

    M, N = p.shape

    print("Poisson (torch-GPU): Computing divergencies from gradient field", flush=True)

    fx = torch.zeros_like(p)
    fy = torch.zeros_like(q)

    fx[:, :-1] = p[:, :-1]
    fy[:-1, :] = q[:-1, :]

    div = (fx - torch.roll(fx, 1, dims=1)) + (fy - torch.roll(fy, 1, dims=0))
    b = div.flatten()

    print("Poisson (torch): Setting up Laplacian operator", flush=True)

    # Costruiamo la matrice Laplaciana come operatore
    # NOTA: per ora solo implementazione matriciale diretta CPU
    Ix = torch.eye(N, device=torch_device)
    Iy = torch.eye(M, device=torch_device)

    eN = torch.ones(N, device=torch_device)
    Dx = torch.diag(-2 * eN) + torch.diag(eN[:-1], diagonal=1) + torch.diag(eN[:-1], diagonal=-1)
    Dx[0, 0] = -1
    Dx[-1, -1] = -1

    eM = torch.ones(M, device=torch_device)
    Dy = torch.diag(-2 * eM) + torch.diag(eM[:-1], diagonal=1) + torch.diag(eM[:-1], diagonal=-1)
    Dy[0, 0] = -1
    Dy[-1, -1] = -1

    A = torch.kron(Iy, Dx) + torch.kron(Dy, Ix)

    print("Poisson (torch-GPU): Solving the system", flush=True)

    # Misura tempo
    t0 = time.time()

    # Risolvi sistema lineare A Z = b
    Z_flat = torch.linalg.solve(A, b)

    elapsed = time.time() - t0
    print(f"Poisson (torch-GPU): Solving time: {elapsed:.2f} seconds")

    Z = Z_flat.reshape((M, N))
    Z -= Z.min()

    return Z.cpu().numpy()

# def depth_from_normal_dct(normals, periodic=False):
#     nx, ny, nz = normals[..., 0], normals[..., 1], normals[..., 2]
#     epsilon = 1e-6
#     p = -nx / (nz + epsilon)
#     q = -ny / (nz + epsilon)
#     Z = depth_from_gradient_dct(p, q, periodic=False)
#     return Z

# def depth_from_gradient_dct(p, q, periodic=False):
#     assert p.shape == q.shape
#     M, N = p.shape
#
#     if not periodic:
#         # Estensione simmetrica
#         p = np.block([
#             [p, -p[:, ::-1]],
#             [p[::-1, :], -p[::-1, ::-1]]
#         ])
#         q = np.block([
#             [q, q[:, ::-1]],
#             [-q[::-1, :], -q[::-1, ::-1]]
#         ])
#         M *= 2
#         N *= 2
#
#     # Divergenza del campo di gradiente
#     dx = np.zeros_like(p)
#     dy = np.zeros_like(q)
#
#     dx[:, :-1] = p[:, :-1]
#     dx[:, -1] = 0  # bordo destro
#     dy[:-1, :] = q[:-1, :]
#     dy[-1, :] = 0  # bordo inferiore
#
#     f = dx - np.roll(dx, 1, axis=1) + dy - np.roll(dy, 1, axis=0)
#
#     # DCT del termine sorgente
#     f_dct = dct2(f)
#
#     # Costruzione degli autovalori del Laplaciano discreto
#     pi = np.pi
#     xx, yy = np.meshgrid(np.arange(N), np.arange(M))
#     denom = (2 * np.cos(pi * xx / N) - 2) + (2 * np.cos(pi * yy / M) - 2)
#     epsilon = 1e-6
#     denom = np.where(denom == 0, epsilon, denom)
#
#     Z_dct = f_dct / denom
#     # Z_dct[0, 0] = 0  # forzare media zero. NO!
#
#     Z = idct2(Z_dct)
#
#     if not periodic:
#         Z = Z[:M//2, :N//2]
#
#     # Level Z to the lowest value
#     offset = np.min(Z)
#     Z = Z - offset
#
#     return Z



def remove_bow_effect(normals, sigma=100, target_mean=np.array([0, 0, 1]), normalize=False):
    """
    Rimuove l'effetto "bombatura" dai bordi della normale stimata.

    Parameters:
        normals: HxWx3 array, con normali tra -1 e 1
        sigma: raggio del filtro gaussiano (maggiore = più aggressivo)
        target_mean: direzione media desiderata (tipicamente [0,0,1])

    Returns:
        Normali corrette, shape (H, W, 3)
    """
    low_freq = gaussian_filter(normals, sigma=(sigma, sigma, 0))

    # HF
    high_freq = normals - low_freq

    # Riallineamento
    corrected = high_freq + target_mean.reshape(1, 1, 3)

    if normalize:
        norm = np.linalg.norm(corrected, axis=2, keepdims=True)
        corrected = corrected / (norm + 1e-8)

    return np.clip(corrected, -1, 1)




def remove_parabolic_trend(Z):
    h, w = Z.shape
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = Z.flatten()

    # Fit suxficie polinomiale grado 3
    poly = PolynomialFeatures(degree=3)
    XY = np.stack([X_flat, Y_flat], axis=1)
    XY_poly = poly.fit_transform(XY)

    model = LinearRegression()
    model.fit(XY_poly, Z_flat)
    Z_fit = model.predict(XY_poly).reshape(h, w)

    # Sottraggo la componente parabolica
    Z_corrected = Z - Z_fit

    #Normalizza
    #Z_corrected = (Z_corrected - np.min(Z_corrected)) / (np.max(Z_corrected) - np.min(Z_corrected))  # da 0 a 1
    return Z_corrected


def recenter_normals(n):
    mean_normal = np.mean(n.reshape(-1, 3), axis=0)
    correction = np.array([0.0, 0.0, 1.0]) - mean_normal
    n = n + correction.reshape(1, 1, 3)
    norm = np.linalg.norm(n, axis=2, keepdims=True)
    return n / (norm + 1e-8)


def normals_from_depth(Z):
    # # Calcolo delle derivate parziali
    # dzdy, dzdx = np.gradient(Z)
    #
    # # Costruzione del vettore normale: [-dz/dx, -dz/dy, 1]
    # n = np.stack([-dzdx, -dzdy, np.ones_like(Z)], axis=-1)
    #
    # # Normalizzazione
    # norm = np.linalg.norm(n, axis=2, keepdims=True)
    # n_normalized = n / (norm + 1e-8)  # X evitare div 0
    # Z = scipy.ndimage.gaussian_filter(Z, sigma=1.0)
    dzdx = cv2.Scharr(Z, cv2.CV_64F, 1, 0)
    dzdy = cv2.Scharr(Z, cv2.CV_64F, 0, 1)
    normals = np.dstack((-dzdx, -dzdy, np.ones_like(Z)))
    normals /= np.linalg.norm(normals, axis=2, keepdims=True)

    return normals



def match_normal_distribution(n_in, n_out):
    """
    Adatta la distribuzione di n_out (normali ricostruite)
    a quella di n_ref (normali originali), mantenendo la direzione.
    """
    n_out_corr = np.copy(n_out)

    for c in range(3):  # x, y, z
        mean_in = np.mean(n_in[..., c])
        std_in = np.std(n_in[..., c])
        mean_out = np.mean(n_out[..., c])
        std_out = np.std(n_out[..., c]) + 1e-8

        # riproporziona canale
        n_out_corr[..., c] = (n_out[..., c] - mean_out) * (std_in / std_out) + mean_in

    # Rinomralizzazione per u
    norm = np.linalg.norm(n_out_corr, axis=2, keepdims=True)
    n_out_corr = n_out_corr / (norm + 1e-8)

    return n_out_corr


# def poisson_solver(fx, fy):
#     """
#     Ricostruisce una superficie Z data la derivata fx = dZ/dx e fy = dZ/dy
#     utilizzando il Poisson Solver via DCT.
#     """
#     h, w = fx.shape
#     # Calcola la divergenza del gradiente (laplaciano approssimato)
#     fxx = np.zeros_like(fx)
#     fyy = np.zeros_like(fy)
#     fxx[:, :-1] = fx[:, :-1] - fx[:, 1:]
#     fyy[:-1, :] = fy[:-1, :] - fy[1:, :]
#     f = fxx + fyy
#
#     # Applica DCT (Discrete Cosine Transform)
#     dct_f = dct(dct(f.T, norm='ortho').T, norm='ortho')
#
#     # Denominatore per il Poisson in frequenza
#     xx, yy = np.meshgrid(np.arange(w), np.arange(h))
#     denom = (2 * np.cos(np.pi * xx / w) - 2) + (2 * np.cos(np.pi * yy / h) - 2)
#     denom[0, 0] = 1  # evita divisione per 0 (DC component)
#     Z = dct_f / denom
#     Z[0, 0] = 0  # fissiamo la media a zero
#
#     # Inversa DCT per ottenere la mappa Z
#     Z = idct(idct(Z.T, norm='ortho').T, norm='ortho')
#     return Z
#
# def depth_from_normals(normals):
#     """
#     Normali devono essere normalizzate, shape: (H, W, 3), valori tra -1 e 1
#     """
#     nz = normals[:, :, 2]
#     nz[nz == 0] = 1e-5  # evita divisioni per zero
#
#     p = -normals[:, :, 0] / nz  # ∂Z/∂x
#     q = -normals[:, :, 1] / nz  # ∂Z/∂y
#
#     Z = poisson_solver(p, q)
#     return Z
#
# def normals_from_depth(Z):
#     dzdy, dzdx = np.gradient(Z)
#     n = np.stack([-dzdx, -dzdy, np.ones_like(Z)], axis=-1)
#     #n /= np.linalg.norm(n, axis=2, keepdims=True)
#     norm = np.linalg.norm(n, axis=2, keepdims=True)
#     n_normalized = n / (norm + 1e-8)
#     return n_normalized



# def depth_from_normals(normals, periodic=False):
#     """
#     Calcola la mappa di profondità da una mappa di normali.
#
#     Parameters:
#         normals: numpy array (H, W, 3), normal map in coordinate (nx, ny, nz)
#         periodic: bool, se True assume condizioni periodiche al bordo
#
#     Returns:
#         Z: numpy array (H, W), mappa di profondità stimata
#     """
#     # Estrai le componenti
#     nx = normals[:, :, 0]
#     ny = normals[:, :, 1]
#     nz = normals[:, :, 2]
#
#     # Evita divisione per zero
#     nz[nz == 0] = 1e-5
#
#     # Calcola i gradienti p = dZ/dx e q = dZ/dy
#     p = -nx / nz
#     q = -ny / nz
#
#     # Applica la funzione di integrazione
#     Z = integrate_gradients(p, q, periodic)
#     return Z
#
#
# def integrate_gradients(p, q, periodic=False):
#     """
#     Integrazione FFT di un campo di gradienti (p, q) in una mappa Z.
#     """
#     M, N = p.shape
#
#     if not periodic:
#         # Crea la versione "mirrorata" per bordi non periodici
#         p = np.block([
#             [p, -p[:, ::-1]],
#             [p[::-1, :], -p[::-1, ::-1]]
#         ])
#         q = np.block([
#             [q, q[:, ::-1]],
#             [-q[::-1, :], -q[::-1, ::-1]]
#         ])
#         M, N = p.shape
#
#     # Frequenze
#     halfM = (M - 1) / 2
#     halfN = (N - 1) / 2
#     u, v = np.meshgrid(np.arange(-np.ceil(halfN), np.floor(halfN) + 1),
#                        np.arange(-np.ceil(halfM), np.floor(halfM) + 1))
#     u = np.fft.ifftshift(u)
#     v = np.fft.ifftshift(v)
#
#     # Trasformata di Fourier
#     Fp = np.fft.fft2(p)
#     Fq = np.fft.fft2(q)
#
#     denom = (u / N) ** 2 + (v / M) ** 2
#     denom[denom == 0] = 1e-5  # evitare divisione per zero
#
#     Fz = -1j / (2 * np.pi) * (u * Fp / N + v * Fq / M) / denom
#     Fz[0, 0] = 0  # componente DC
#
#     Z = np.real(np.fft.ifft2(Fz))
#
#     if not periodic:
#         Z = Z[:M // 2, :N // 2]
#
#     return Z