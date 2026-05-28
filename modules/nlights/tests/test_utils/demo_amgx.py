import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import os
import pyamgx
import time
from utils.depth_estimation import  depth_from_gradient_poisson_gpu_pyamgx


# read : ("input_file", ["normals.npy"]) 
input_file = "normals.npy"
# Load input data
normals = np.load(input_file)
# Compute depth maps using different methods
cuda_start = time.time()
# depth_fullcuda = compute_depth_maps(normals, method='fullcuda')
cfg_test = r'''{
    "config_version": 2,
    "determinism_flag": 1,
    "exception_handling": 1,
    "solver": {
        "solver": "PCG",
        "print_solve_stats": 1,
        "max_iters": 5000,
        "tolerance": 1e-6,
        "norm": "L2",
        "convergence": "RELATIVE_INI_CORE",
        "monitor_residual": 1,
        "preconditioner": {
            "solver": "AMG",
            "algorithm": "AGGREGATION",
            "max_levels": 5,
            "presweeps": 1,
            "postsweeps": 1,
            "interpolator": "D2",
            "selector": "SIZE_8",
            "cycle": "V",
            "relaxation_factor": 0.75
        }
    }
}'''
depth_fullcuda = depth_from_gradient_poisson_gpu_pyamgx(normals, config_string=cfg_test)
cuda_end = time.time()
print(f"AMGX Time: {cuda_end - cuda_start:.6f} seconds")
np.save("depth_fullcuda.npy", depth_fullcuda)
pyamgx.initialize()
cfg_test = r'''
{
    "config_version": 2,
        "determinism_flag": 1,
        "exception_handling" : 1,
        "solver": {
            "monitor_residual": 1,
            "solver": "BICGSTAB",
            "convergence": "RELATIVE_INI_CORE",
            "preconditioner": {
                "solver": "NOSOLVER"
        }
    }
}
'''
cfg_test = r'''{
    "config_version": 2,
    "determinism_flag": 1,
    "exception_handling": 1,
    "solver": {
        "solver": "PCG",
        "print_solve_stats": 1,
        "max_iters": 5000,
        "tolerance": 1e-6,
        "norm": "L2",
        "convergence": "RELATIVE_INI_CORE",
        "monitor_residual": 1,
        "preconditioner": {
            "solver": "AMG",
            "algorithm": "AGGREGATION",
            "max_levels": 5,
            "presweeps": 1,
            "postsweeps": 1,
            "interpolator": "D2",
            "selector": "SIZE_8",
            "cycle": "V",
            "relaxation_factor": 0.75
        }
    }
}'''

# Initialize config and resources:
# os get full path of the file "pyamgx_config.json"
cfg = pyamgx.Config().create(cfg_test)
rsc = pyamgx.Resources().create_simple(cfg)
pyamgx.finalize()

exit(0)
# Create matrices and vectors:
A = pyamgx.Matrix().create(rsc)
b = pyamgx.Vector().create(rsc)
x = pyamgx.Vector().create(rsc)

# Create solver:
solver = pyamgx.Solver().create(rsc, cfg)

# Upload system:

M = sparse.csr_matrix(np.random.rand(5, 5))
rhs = np.random.rand(5)
sol = np.zeros(5, dtype=np.float64)

A.upload_CSR(M)
b.upload(rhs)
x.upload(sol)

# Setup and solve system:
solver.setup(A)
solver.solve(b, x)

# Download solution
x.download(sol)
print("pyamgx solution: ", sol)
print("scipy solution: ", splinalg.spsolve(M, rhs))

# Clean up:
A.destroy()
x.destroy()
b.destroy()
solver.destroy()
rsc.destroy()
cfg.destroy()

pyamgx.finalize()
