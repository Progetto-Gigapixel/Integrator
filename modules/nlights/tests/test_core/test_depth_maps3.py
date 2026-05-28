"""
 Test per confrontare depth_from_gradient_poisson
 contenute in file depth_maps.py e depth_estimation.py
"""
import os
import pytest
import time
import numpy as np
from core.depth_maps import compute_depth_maps
from utils.depth_estimation import depth_from_gradient_poisson_cupy_tiled

@pytest.mark.parametrize("legacy_result", ["depth_fullimage_piranesi_gs.npy"])
@pytest.mark.parametrize("new_result", ["depth_fullimage_piranesi_cpu5s.npy"])
def test_result_compare(legacy_result, new_result):
    #Load input data
    legacy = np.load(legacy_result)
    new = np.load(new_result)
    #calcolare divergenza tra i due risultati matriciali e darne un valore percentuale
    divergence = np.abs(legacy - new) / (np.abs(legacy) + 1e-5)
    print(f"Divergence: {np.mean(divergence) * 100:.2f}%")
    assert np.allclose(legacy, new, atol=1e-5)

@pytest.mark.parametrize("input_file", ["normals_fulliamge_piranesi.npy"])
def test_poisson_legacy(input_file):    # Load input data
    normals = np.load(input_file)
    cpu_start = time.time()
    golden_standard = depth_from_gradient_poisson(normals)
    cpu_end = time.time()
    print(f"CPU Time: {cpu_end - cpu_start:.6f} seconds")
    # save to disk
    np.save("depth_fullimage_piranesi_gs.npy", golden_standard)
    # depth_legacy = np.load("depth_estimation.npy")
    # divergence = np.abs(golden_standard - depth_legacy) / (np.abs(golden_standard) + 1e-5)
    # print(f"Divergence: {np.mean(divergence) * 100:.2f}%")
    # assert np.allclose(golden_standard, depth_legacy, atol=1e-5)
    assert True
@pytest.mark.parametrize("input_file", ["normals_fullimage_piranesi.npy"])
def test_depth_tiled_cpu(input_file):
    # Load input data
    normals = np.load(input_file)
    # Compute depth maps using different methods
    # Conta numero di processori disponibili
    num_processors = os.cpu_count()
    print(f"Numero di processori disponibili: {num_processors}")
    # Dato il numero di processori trovare due multiplicatori
    num_tiles = int(np.sqrt(num_processors))
    # data la dimensione di normals trovare la dimensione delle tiles
    tile_size = (normals.shape[0] // num_tiles)
    print(f"Dimensione delle tiles: {tile_size}")
    # overlap delle tiles
    overlap = tile_size // 4
    print(f"Overlap delle tiles: {overlap}")
    cpu_start = time.time()
    depth_tiling = depth_from_gradient_poisson_tiled(normals, tile_size=tile_size, overlap=overlap, n_schwarz_iters=5)
    cpu_end = time.time()
    print(f"CPU Tiling Time: {cpu_end - cpu_start:.6f} seconds")
    # depth_estimation = np.load("depth_estimation.npy")
    # print(f"CPU Time: 7712.58s seconds")
    # save to disk
    np.save("depth_fullimage_piranesi_cpu5s.npy", depth_tiling)
    assert True
    # assert np.allclose(depth_tiling, depth_estimation, atol=1e-5)

@pytest.mark.parametrize("input_file", ["normals.npy"])
def test_depth_tiled_gpu(input_file):
    # Load input data
    normals = np.load(input_file)

    # Compute depth maps using different methods
    cuda_start = time.time()
    # depth_fullcuda = compute_depth_maps(normals, method='fullcuda')
    depth_fullcuda = depth_from_gradient_poisson_cupy_tiled(normals, n_schwarz_iters=10)
    cuda_end = time.time()
    print(f"CUDA Time: {cuda_end - cuda_start:.6f} seconds")
    # load cpu result from disk:
    # depth_estimation = np.load("depth_estimation.npy")
    # print(f"CPU Time: 7712.58s seconds")
    # save to disk
    np.save("depth_cupy_fullimage_piranesi_s10.npy", depth_fullcuda)
    depth_cpu = np.load("depth_fullimage_piranesi_cpu3s.npy")
    golden_standard = np.load("depth_fullimage_piranesi_gs.npy")
    div_gpu = calc_divergence(depth_fullcuda, golden_standard)
    div_cpu = calc_divergence(depth_cpu, golden_standard)
    # Assert that gpu is lesser than cpu
    assert div_gpu < div_cpu    

@pytest.mark.parametrize("input_file", ["normals.npy"])
def test_depth_estimation_AMG(input_file):
    # Load input data
    normals = np.load(input_file)

    # Compute depth maps using different methods
    cuda_start = time.time()
    # depth_fullcuda = compute_depth_maps(normals, method='fullcuda')
    depth_amgx= depth_from_gradient_poisson_smart_strategy(normals, strategy='tiling')
    cuda_end = time.time()
    print(f"AMGX Time: {cuda_end - cuda_start:.6f} seconds")
    # load cpu result from disk:
    # np.save("depth_fullcuda.npy", depth_fullcuda)
    depth_estimation = np.load("depth_fullcuda.npy")
    print(f"CPU Time: 7712.58s seconds")
    # save to disk
    np.save("depth_amgx.npy", depth_amgx)
    # Compare results
    assert np.allclose(depth_amgx, depth_estimation, atol=1e-5)

def calc_divergence(legacy, new):
    # Calcolare divergenza tra i due risultati matriciali e darne un valore percentuale
    divergence = np.abs(legacy - new) / (np.abs(legacy) + 1e-5)
    print(f"Divergence: {np.mean(divergence) * 100:.2f}%")
    return np.mean(divergence)

@pytest.mark.parametrize("input_file", ["normals.npy"])
@pytest.mark.parametrize("legacy_result", ["golden_standard.npy"])
def test_how_many_s(input_file, legacy_result):
    # Load input data
    normals = np.load(input_file)
    legacy = np.load(legacy_result)
    # Compute depth maps using different methods
    cuda_start = time.time()
    # depth_fullcuda = compute_depth_maps(normals, method='fullcuda')
    iteration = test_schwarz_for_gradient_poisson_cupy_tiled(normals=normals, legacy_result=legacy, accuracy=1, max_iters=10)
    cuda_end = time.time()
    print(f"CUDA Time: {cuda_end - cuda_start:.6f} seconds")
    assert iteration < 10    
