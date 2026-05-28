"""
Functions for computing depth maps from normal maps.
Python implementation of DepthFromGradient.m
"""

import numpy as np
# import scipy
from scipy import sparse
from scipy.sparse.linalg import spsolve as spsolve_scipy
import numba as nb
from scipy import sparse as sp
import cupy as cp
from cupyx.scipy import sparse as csp
from cupyx.scipy.sparse.linalg import spsolve as spsolve_cuda

def compute_depth_maps(normal_map, mask=None, method='poisson', progress_callback=None):
    """
    Compute depth map from normal map using gradient integration.
    
    Parameters
    ----------
    normal_map : ndarray
        Normal map, shape (height, width, 3) with values in [-1, 1]
    mask : ndarray, optional
        Binary mask of valid pixels, shape (height, width)
    method : str, optional
        Method for depth reconstruction: 'poisson', 'frankot', or 'direct'
    progress_callback : function, optional
        Function to call with progress updates (0-100)
        
    Returns
    -------
    ndarray
        Depth map, shape (height, width)
    """
    # Update progress
    if progress_callback:
        progress_callback(0)
    
    # Extract p and q components from normals (gradient fields)
    # p = -n_x/n_z, q = -n_y/n_z
    p = -normal_map[:, :, 0] / np.clip(normal_map[:, :, 2], 1e-8, None)
    q = -normal_map[:, :, 1] / np.clip(normal_map[:, :, 2], 1e-8, None)
    
    if progress_callback:
        progress_callback(10)
    
    # Apply selected integration method
    if method == 'poisson':
        depth_map = depth_from_gradient_poisson(p, q, mask, progress_callback)
    elif method == 'frankot':
        depth_map = depth_from_gradient_frankot(p, q, mask, progress_callback)
    elif method == 'direct':
        depth_map = depth_from_gradient_direct(p, q, mask, progress_callback)
    elif method == 'fullcuda':
        depth_map = depth_from_gradient_poisson_fullcuda(p, q, mask, progress_callback)
    elif method == 'hybrid':
        depth_map = depth_from_gradient_poisson_hybrid(p, q, mask, progress_callback)
    elif method == 'vectorized':
        depth_map = depth_from_gradient_poisson_vectorized(p, q, mask, progress_callback)
    else:
        raise ValueError(f"Unknown depth reconstruction method: {method}")
    
    if progress_callback:
        progress_callback(100)
    
    return depth_map


def depth_from_gradient_poisson(p, q, mask=None, progress_callback=None):
    """
    Reconstruct depth map from gradient fields using Poisson integration.
    
    Parameters
    ----------
    p : ndarray
        Gradient in x direction, shape (height, width)
    q : ndarray
        Gradient in y direction, shape (height, width)
    mask : ndarray, optional
        Binary mask of valid pixels, shape (height, width)
    progress_callback : function, optional
        Function to call with progress updates (0-100)
        
    Returns
    -------
    ndarray
        Depth map, shape (height, width)
    """
    height, width = p.shape
    
    # If no mask provided, use all pixels
    if mask is None:
        mask = np.ones((height, width), dtype=bool)
    
    if progress_callback:
        progress_callback(20)
    
    # Construct the 2D Laplacian operator
    # Each pixel is connected to its 4 neighbors with weight -1 and to itself with weight 4
    row_indices = []
    col_indices = []
    values = []
    
    # Create mapping from (row, col) to linear index
    pixel_indices = -np.ones((height, width), dtype=int)
    valid_pixels = np.where(mask)
    num_valid_pixels = len(valid_pixels[0])
    pixel_indices[valid_pixels] = np.arange(num_valid_pixels)
    
    # Helper function to add matrix entries
    def add_entry(i_row, i_col, value):
        row_indices.append(i_row)
        col_indices.append(i_col)
        values.append(value)
    
    # Compute divergence of gradient field: div(p,q) = dp/dx + dq/dy
    # We'll use central differences for the divergence
    divergence = np.zeros((height, width))
    
    # dp/dx
    divergence[:, 1:-1] += (p[:, 2:] - p[:, :-2]) / 2.0
    # dq/dy
    divergence[1:-1, :] += (q[2:, :] - q[:-2, :]) / 2.0
    
    # Construct the sparse linear system: L*z = divergence
    # where L is the Laplacian operator
    for i in range(height):
        for j in range(width):
            if not mask[i, j]:
                continue
            
            # Current pixel index
            idx = pixel_indices[i, j]
            
            # Add center pixel
            neighbors = 0
            
            # Add neighbor entries if in bounds and masked
            if i > 0 and mask[i-1, j]:
                add_entry(idx, pixel_indices[i-1, j], -1)
                neighbors += 1
            
            if i < height-1 and mask[i+1, j]:
                add_entry(idx, pixel_indices[i+1, j], -1)
                neighbors += 1
            
            if j > 0 and mask[i, j-1]:
                add_entry(idx, pixel_indices[i, j-1], -1)
                neighbors += 1
            
            if j < width-1 and mask[i, j+1]:
                add_entry(idx, pixel_indices[i, j+1], -1)
                neighbors += 1
            
            # Add diagonal entry
            add_entry(idx, idx, neighbors)
    
    if progress_callback:
        progress_callback(50)
    
    # Create sparse matrix and solve the linear system
    laplacian = sparse.coo_matrix((values, (row_indices, col_indices)), 
                                 shape=(num_valid_pixels, num_valid_pixels))
    laplacian = laplacian.tocsr()
    
    # Right-hand side: divergence for all valid pixels
    b = divergence[mask]
    
    # Solve the linear system
    z_values = spsolve_scipy(laplacian, b)
    
    if progress_callback:
        progress_callback(90)
    
    # Create and fill output depth map
    z = np.zeros((height, width))
    z[mask] = z_values
    
    return z

def depth_from_gradient_poisson_vectorized(p, q, mask=None, progress_callback=None):
    """
    Reconstruct depth map from gradient fields using Poisson integration and vectorization.
    This is a vectorized version of the Poisson integration method, running on CPU.
    Parameters
    ----------
    p : ndarray
        Gradient in x direction, shape (height, width)
    q : ndarray
        Gradient in y direction, shape (height, width)
    mask : ndarray, optional
        Binary mask of valid pixels, shape (height, width)
    progress_callback : function, optional
        Function to call with progress updates (0-100)
        
    Returns
    -------
    ndarray
        Depth map, shape (height, width)
    """
    height, width = p.shape

    if mask is None:
        mask = np.ones((height, width), dtype=bool)

    if progress_callback:
        progress_callback(20)

    pixel_indices = -np.ones((height, width), dtype=np.int32)
    valid_pixels = np.where(mask)
    num_valid_pixels = len(valid_pixels[0])
    pixel_indices[valid_pixels] = np.arange(num_valid_pixels, dtype=np.int32)

    # Divergence
    divergence = np.zeros((height, width), dtype=np.float32)
    divergence[:, 1:-1] += (p[:, 2:] - p[:, :-2]) * 0.5
    divergence[1:-1, :] += (q[2:, :] - q[:-2, :]) * 0.5

    if progress_callback:
        progress_callback(50)

    # Build Laplacian on CPU with Numba
        
    # Vettorializzazione della costruzione della matrice Laplaciana
    row_indices = []
    col_indices = []
    values = []

    # Aggiungi le voci per i vicini a destra
    right_mask = np.zeros_like(mask)
    right_mask[:, :-1] = mask[:, 1:]
    valid_right_neighbors = mask & right_mask
    rows, cols = np.where(valid_right_neighbors)
    row_indices.extend(pixel_indices[rows, cols].tolist())
    col_indices.extend(pixel_indices[rows, cols + 1].tolist())
    values.extend([-1.0] * len(rows))

    # Aggiungi le voci per i vicini a sinistra
    left_mask = np.zeros_like(mask)
    left_mask[:, 1:] = mask[:, :-1]
    valid_left_neighbors = mask & left_mask
    rows, cols = np.where(valid_left_neighbors)
    row_indices.extend(pixel_indices[rows, cols].tolist())
    col_indices.extend(pixel_indices[rows, cols - 1].tolist())
    values.extend([-1.0] * len(rows))

    # Aggiungi le voci per i vicini sopra
    up_mask = np.zeros_like(mask)
    up_mask[1:, :] = mask[:-1, :]
    valid_up_neighbors = mask & up_mask
    rows, cols = np.where(valid_up_neighbors)
    row_indices.extend(pixel_indices[rows, cols].tolist())
    col_indices.extend(pixel_indices[rows - 1, cols].tolist())
    values.extend([-1.0] * len(rows))

    # Aggiungi le voci per i vicini sotto
    down_mask = np.zeros_like(mask)
    down_mask[:-1, :] = mask[1:, :]
    valid_down_neighbors = mask & down_mask
    rows, cols = np.where(valid_down_neighbors)
    row_indices.extend(pixel_indices[rows, cols].tolist())
    col_indices.extend(pixel_indices[rows + 1, cols].tolist())
    values.extend([-1.0] * len(rows))
    
    # Calcola il numero di vicini validi per ogni pixel e aggiungi le voci diagonali
    num_neighbors = (valid_right_neighbors + valid_left_neighbors + valid_up_neighbors + valid_down_neighbors)
    row_indices.extend(pixel_indices[mask].tolist())
    col_indices.extend(pixel_indices[mask].tolist())
    values.extend(num_neighbors[mask].tolist())

    laplacian = sparse.coo_matrix((values, (row_indices, col_indices)),
                                  shape=(num_valid_pixels, num_valid_pixels)).tocsr()
    
    b_cpu = divergence[mask].astype(np.float32)

    if progress_callback:
        progress_callback(70)
    
    laplacian = sparse.coo_matrix((values, (row_indices, col_indices)),
                                  shape=(num_valid_pixels, num_valid_pixels)).tocsr()
    
    b_cpu = divergence[mask].astype(np.float32)

    if progress_callback:
        progress_callback(70)

    # Right-hand side: divergence for all valid pixels
    b = divergence[mask]
    
    # Solve the linear system
    z_values = spsolve_scipy(laplacian, b)
    
    if progress_callback:
        progress_callback(90)
    
    # Create and fill output depth map
    z = np.zeros((height, width))
    z[mask] = z_values
        
    if progress_callback:
        progress_callback(100)
    return z


def depth_from_gradient_poisson_hybrid(p, q, mask=None, progress_callback=None):
    height, width = p.shape

    if mask is None:
        mask = np.ones((height, width), dtype=bool)

    if progress_callback:
        progress_callback(20)

    pixel_indices = -np.ones((height, width), dtype=np.int32)
    valid_pixels = np.where(mask)
    num_valid_pixels = len(valid_pixels[0])
    pixel_indices[valid_pixels] = np.arange(num_valid_pixels, dtype=np.int32)

    # Divergence
    divergence = np.zeros((height, width), dtype=np.float32)
    divergence[:, 1:-1] += (p[:, 2:] - p[:, :-2]) * 0.5
    divergence[1:-1, :] += (q[2:, :] - q[:-2, :]) * 0.5

    if progress_callback:
        progress_callback(50)

    # Build Laplacian on CPU with Numba
        
    # Vettorializzazione della costruzione della matrice Laplaciana
    row_indices = []
    col_indices = []
    values = []

    # Aggiungi le voci per i vicini a destra
    right_mask = np.zeros_like(mask)
    right_mask[:, :-1] = mask[:, 1:]
    valid_right_neighbors = mask & right_mask
    rows, cols = np.where(valid_right_neighbors)
    row_indices.extend(pixel_indices[rows, cols].tolist())
    col_indices.extend(pixel_indices[rows, cols + 1].tolist())
    values.extend([-1.0] * len(rows))

    # Aggiungi le voci per i vicini a sinistra
    left_mask = np.zeros_like(mask)
    left_mask[:, 1:] = mask[:, :-1]
    valid_left_neighbors = mask & left_mask
    rows, cols = np.where(valid_left_neighbors)
    row_indices.extend(pixel_indices[rows, cols].tolist())
    col_indices.extend(pixel_indices[rows, cols - 1].tolist())
    values.extend([-1.0] * len(rows))

    # Aggiungi le voci per i vicini sopra
    up_mask = np.zeros_like(mask)
    up_mask[1:, :] = mask[:-1, :]
    valid_up_neighbors = mask & up_mask
    rows, cols = np.where(valid_up_neighbors)
    row_indices.extend(pixel_indices[rows, cols].tolist())
    col_indices.extend(pixel_indices[rows - 1, cols].tolist())
    values.extend([-1.0] * len(rows))

    # Aggiungi le voci per i vicini sotto
    down_mask = np.zeros_like(mask)
    down_mask[:-1, :] = mask[1:, :]
    valid_down_neighbors = mask & down_mask
    rows, cols = np.where(valid_down_neighbors)
    row_indices.extend(pixel_indices[rows, cols].tolist())
    col_indices.extend(pixel_indices[rows + 1, cols].tolist())
    values.extend([-1.0] * len(rows))
    
    # Calcola il numero di vicini validi per ogni pixel e aggiungi le voci diagonali
    num_neighbors = (valid_right_neighbors + valid_left_neighbors + valid_up_neighbors + valid_down_neighbors)
    row_indices.extend(pixel_indices[mask].tolist())
    col_indices.extend(pixel_indices[mask].tolist())
    values.extend(num_neighbors[mask].tolist())

    laplacian = sparse.coo_matrix((values, (row_indices, col_indices)),
                                  shape=(num_valid_pixels, num_valid_pixels)).tocsr()
    
    b_cpu = divergence[mask].astype(np.float32)

    if progress_callback:
        progress_callback(70)
    
    laplacian_cpu = sparse.coo_matrix((values, (row_indices, col_indices)),
                                  shape=(num_valid_pixels, num_valid_pixels)).tocsr()
    
    b_cpu = divergence[mask].astype(np.float32)

    if progress_callback:
        progress_callback(70)

    # Move to GPU
    laplacian_gpu = csp.csr_matrix(laplacian_cpu)
    b_gpu = cp.asarray(b_cpu)

    # Solve in GPU
    z_values_gpu = spsolve_cuda(laplacian_gpu, b_gpu)

    if progress_callback:
        progress_callback(90)

    # Back to CPU
    z = np.zeros((height, width), dtype=np.float32)
    z[mask] = cp.asnumpy(z_values_gpu)
    if progress_callback:
        progress_callback(100)
    return z

import cupy as cp
from cupyx.scipy import sparse as csp
from cupyx.scipy.sparse.linalg import spsolve as spsolve_cuda
from cupyx.scipy.sparse.linalg import cg as cg_cuda


laplacian_kernel = cp.RawKernel(r'''
extern "C" __global__
void build_laplacian(const int height, const int width,
                     const bool* __restrict__ mask,
                     const int* __restrict__ pixel_indices,
                     int* __restrict__ row_indices,
                     int* __restrict__ col_indices,
                     float* __restrict__ values) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int total = height * width;
    if (tid >= total) return;

    int i = tid / width;
    int j = tid % width;
    if (!mask[tid]) return;

    int idx = pixel_indices[tid];
    int pos = idx * 5; // max 5 entries per row (4 neighbors + diag)

    int neighbors = 0;
    if (i > 0 && mask[(i-1)*width + j]) {
        row_indices[pos] = idx;
        col_indices[pos] = pixel_indices[(i-1)*width + j];
        values[pos] = -1.0f;
        pos++;
        neighbors++;
    }
    if (i < height-1 && mask[(i+1)*width + j]) {
        row_indices[pos] = idx;
        col_indices[pos] = pixel_indices[(i+1)*width + j];
        values[pos] = -1.0f;
        pos++;
        neighbors++;
    }
    if (j > 0 && mask[i*width + (j-1)]) {
        row_indices[pos] = idx;
        col_indices[pos] = pixel_indices[i*width + (j-1)];
        values[pos] = -1.0f;
        pos++;
        neighbors++;
    }
    if (j < width-1 && mask[i*width + (j+1)]) {
        row_indices[pos] = idx;
        col_indices[pos] = pixel_indices[i*width + (j+1)];
        values[pos] = -1.0f;
        pos++;
        neighbors++;
    }
    // Diagonal entry
    row_indices[pos] = idx;
    col_indices[pos] = idx;
    values[pos] = (float)neighbors;
}
''', 'build_laplacian')

def depth_from_gradient_poisson_fullcuda(p, q, mask=None, progress_callback=None):
    height, width = p.shape

    # --- CPU to GPU ---
    p_gpu = cp.asarray(p, dtype=cp.float32)
    q_gpu = cp.asarray(q, dtype=cp.float32)
    if mask is None:
        mask_gpu = cp.ones((height, width), dtype=cp.bool_)
    else:
        mask_gpu = cp.asarray(mask, dtype=cp.bool_)

    if progress_callback:
        progress_callback(20)

    # Compute pixel indices (GPU)
    pixel_indices_gpu = cp.full((height, width), -1, dtype=cp.int32)
    valid_pixels = cp.argwhere(mask_gpu.ravel()).ravel()
    num_valid_pixels = valid_pixels.size
    pixel_indices_gpu.ravel()[mask_gpu.ravel()] = cp.arange(num_valid_pixels, dtype=cp.int32)

    # Divergence (GPU)
    divergence_gpu = cp.zeros((height, width), dtype=cp.float32)
    divergence_gpu[:, 1:-1] += (p_gpu[:, 2:] - p_gpu[:, :-2]) * 0.5
    divergence_gpu[1:-1, :] += (q_gpu[2:, :] - q_gpu[:-2, :]) * 0.5

    if progress_callback:
        progress_callback(50)

    # Allocate COO arrays (max 5 entries per pixel)
    max_entries = num_valid_pixels * 5
    row_idx_gpu = cp.full((max_entries,), -1, dtype=cp.int32)
    col_idx_gpu = cp.full((max_entries,), -1, dtype=cp.int32)
    vals_gpu = cp.zeros((max_entries,), dtype=cp.float32)

    # Kernel launch
    threads = 256
    blocks = (height * width + threads - 1) // threads
    laplacian_kernel((blocks,), (threads,),
                     (height, width,
                      mask_gpu.ravel(),
                      pixel_indices_gpu.ravel(),
                      row_idx_gpu,
                      col_idx_gpu,
                      vals_gpu))

    # Filtra valori validi (alcuni slot restano -1)
    valid_mask = row_idx_gpu >= 0
    row_idx_gpu = row_idx_gpu[valid_mask]
    col_idx_gpu = col_idx_gpu[valid_mask]
    vals_gpu = vals_gpu[valid_mask]

    # Crea matrice sparsa CSR in GPU
    laplacian_gpu = csp.coo_matrix((vals_gpu, (row_idx_gpu, col_idx_gpu)),
                                   shape=(num_valid_pixels, num_valid_pixels)).tocsr()

    b_gpu = divergence_gpu[mask_gpu].astype(cp.float32)

    # Risolvi su GPU
    z_values_gpu = cg_cuda(laplacian_gpu, b_gpu)

    if progress_callback:
        progress_callback(90)

    # Output CPU
    z = cp.zeros((height, width), dtype=cp.float32)
    z[mask_gpu] = z_values_gpu
    
    # Cleanup GPU memory
    del p_gpu, q_gpu, mask_gpu, pixel_indices_gpu, divergence_gpu
    del row_idx_gpu, col_idx_gpu, vals_gpu, laplacian_gpu, b_gpu, z_values_gpu
    cp.get_default_memory_pool().free_all_blocks()
    cp.cuda.Device().synchronize()
    return cp.asnumpy(z)

def depth_from_gradient_frankot(p, q, mask=None, progress_callback=None):
    """
    Reconstruct depth map from gradient fields using Frankot-Chellappa algorithm.
    
    Parameters
    ----------
    p : ndarray
        Gradient in x direction, shape (height, width)
    q : ndarray
        Gradient in y direction, shape (height, width)
    mask : ndarray, optional
        Binary mask of valid pixels, shape (height, width)
    progress_callback : function, optional
        Function to call with progress updates (0-100)
        
    Returns
    -------
    ndarray
        Depth map, shape (height, width)
    """
    height, width = p.shape
    
    # If no mask provided, use all pixels
    if mask is None:
        mask = np.ones((height, width), dtype=bool)
    
    # Apply mask to gradient fields
    p = p.copy()
    q = q.copy()
    p[~mask] = 0
    q[~mask] = 0
    
    if progress_callback:
        progress_callback(30)
    
    # Compute FFT of gradient fields
    p_fft = np.fft.fft2(p)
    q_fft = np.fft.fft2(q)
    
    if progress_callback:
        progress_callback(50)
    
    # Compute frequencies
    u = np.fft.fftfreq(width).reshape(1, width)
    v = np.fft.fftfreq(height).reshape(height, 1)
    
    # Avoid division by zero
    denom = 2j * np.pi * (u + 1j * v)
    denom[0, 0] = 1  # Avoid division by zero
    
    # Integrate in frequency domain
    z_fft = (-1j * p_fft * 2 * np.pi * u - 1j * q_fft * 2 * np.pi * v) / (4 * np.pi**2 * (u**2 + v**2))
    z_fft[0, 0] = 0  # Set DC component to 0
    
    if progress_callback:
        progress_callback(80)
    
    # Compute inverse FFT to get depth map
    z = np.real(np.fft.ifft2(z_fft))
    
    # Apply mask to result
    z[~mask] = 0
    
    return z

def depth_from_gradient_direct(p, q, mask=None, progress_callback=None):
    """
    Simple direct integration of gradient fields.
    This is a basic implementation and may not produce optimal results.
    
    Parameters
    ----------
    p : ndarray
        Gradient in x direction, shape (height, width)
    q : ndarray
        Gradient in y direction, shape (height, width)
    mask : ndarray, optional
        Binary mask of valid pixels, shape (height, width)
    progress_callback : function, optional
        Function to call with progress updates (0-100)
        
    Returns
    -------
    ndarray
        Depth map, shape (height, width)
    """
    height, width = p.shape
    
    # If no mask provided, use all pixels
    if mask is None:
        mask = np.ones((height, width), dtype=bool)
    
    if progress_callback:
        progress_callback(20)
    
    # Initialize depth map
    z = np.zeros((height, width))
    
    # Start from the center and integrate outward
    center_y, center_x = height // 2, width // 2
    
    # Make sure center is in mask
    if not mask[center_y, center_x]:
        # Find the nearest masked pixel
        valid_pixels = np.where(mask)
        if len(valid_pixels[0]) > 0:
            distances = (valid_pixels[0] - center_y)**2 + (valid_pixels[1] - center_x)**2
            nearest_idx = np.argmin(distances)
            center_y, center_x = valid_pixels[0][nearest_idx], valid_pixels[1][nearest_idx]
        else:
            # No valid pixels, return zeros
            return z
    
    if progress_callback:
        progress_callback(30)
    
    # For simplicity, we'll integrate along rows and columns from the center
    # First, integrate horizontally from center
    for j in range(center_x+1, width):
        if mask[center_y, j] and mask[center_y, j-1]:
            z[center_y, j] = z[center_y, j-1] + p[center_y, j-1]
    
    for j in range(center_x-1, -1, -1):
        if mask[center_y, j] and mask[center_y, j+1]:
            z[center_y, j] = z[center_y, j+1] - p[center_y, j]
    
    if progress_callback:
        progress_callback(50)
    
    # Then integrate vertically from the horizontal line
    for i in range(center_y+1, height):
        for j in range(width):
            if mask[i, j] and mask[i-1, j]:
                z[i, j] = z[i-1, j] + q[i-1, j]
    
    for i in range(center_y-1, -1, -1):
        for j in range(width):
            if mask[i, j] and mask[i+1, j]:
                z[i, j] = z[i+1, j] - q[i, j]
    
    if progress_callback:
        progress_callback(90)
    
    # Apply mask to result
    z[~mask] = 0
    
    return z


def apply_poly_correction(depth_map, mask=None, degree=2):
    """
    Apply polynomial correction to the depth map.
    This helps remove low-frequency noise and global warping.
    
    Parameters
    ----------
    depth_map : ndarray
        Input depth map, shape (height, width)
    mask : ndarray, optional
        Binary mask of valid pixels, shape (height, width)
    degree : int, optional
        Degree of polynomial surface to fit
        
    Returns
    -------
    ndarray
        Corrected depth map, shape (height, width)
    """
    height, width = depth_map.shape
    
    # If no mask provided, use all pixels
    if mask is None:
        mask = np.ones((height, width), dtype=bool)
    
    # Create coordinate grids
    y, x = np.mgrid[:height, :width]
    
    # Valid coordinates and depths
    valid = mask > 0
    x_valid = x[valid]
    y_valid = y[valid]
    z_valid = depth_map[valid]
    
    # Create polynomial terms
    terms = []
    for i in range(degree+1):
        for j in range(degree+1-i):
            terms.append(x_valid**i * y_valid**j)
    
    # Stack terms to create design matrix
    A = np.column_stack(terms)
    
    # Solve for polynomial coefficients
    coeffs, residuals, rank, s = np.linalg.lstsq(A, z_valid, rcond=None)
    
    # Create polynomial surface
    surface = np.zeros_like(depth_map)
    
    # Apply coefficients to create surface
    term_idx = 0
    for i in range(degree+1):
        for j in range(degree+1-i):
            surface += coeffs[term_idx] * (x**i) * (y**j)
            term_idx += 1
    
    # Subtract fitted surface from depth map
    corrected = depth_map - surface
    
    # Apply mask
    corrected[~valid] = 0
    
    return corrected
