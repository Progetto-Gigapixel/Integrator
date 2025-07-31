import numpy as np
import pyceres
import logging
import datetime
from collections import defaultdict
now = datetime.datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
logger = logging.getLogger("CeresBA")
from typing import List, Tuple

#logging.basicConfig(filename=f'CeresBA-{timestamp}.log', level=logging.INFO)

class AffineReprojectionError(pyceres.CostFunction):
    def __init__(self, x1, y1, x2, y2):
        super().__init__()
        # scambio i punti
        # To-do soluzione temporanea
        self.x2, self.y2, self.x1, self.y1 = x1, y1, x2, y2

        # 2 residuals (x, y), each affine has 6 parameters
        self.set_num_residuals(2)
        self.set_parameter_block_sizes([6, 6])  # Affine has 6 parameters per matrix

    def Evaluate(self, parameters, residuals, jacobians):
        A1 = np.array(parameters[0]).reshape(2, 3)
        A2 = np.array(parameters[1]).reshape(2, 3)
        # Transform points
        p1 = np.array([self.x1, self.y1, 1.0])
        p2 = np.array([self.x2, self.y2, 1.0])
        p1_transformed = A1 @ p1
        p2_transformed = A2 @ p2

        # Residuals: reprojection error
        residuals[0] = p1_transformed[0] - p2_transformed[0]
        residuals[1] = p1_transformed[1] - p2_transformed[1]
        x, y = self.x1, self.y1
        # Jacobians
        if jacobians is not None:
            if jacobians[0] is not None:
                # Derivative w.r.t A1 (positive)
                jacobians[0][0] = x
                jacobians[0][1] = y
                jacobians[0][2] = 1.0
                jacobians[0][3] = 0.0
                jacobians[0][4] = 0.0
                jacobians[0][5] = 0.0

                jacobians[0][6] = 0.0
                jacobians[0][7] = 0.0
                jacobians[0][8] = 0.0
                jacobians[0][9] = x
                jacobians[0][10] = y
                jacobians[0][11] = 1.0

            if jacobians[1] is not None:
                # Derivative w.r.t A2 (negative)
                jacobians[1][0] = -x
                jacobians[1][1] = -y
                jacobians[1][2] = -1.0
                jacobians[1][3] = 0.0
                jacobians[1][4] = 0.0
                jacobians[1][5] = 0.0

                jacobians[1][6] = 0.0
                jacobians[1][7] = 0.0
                jacobians[1][8] = 0.0
                jacobians[1][9] = -x
                jacobians[1][10] = -y
                jacobians[1][11] = -1.0

        return True

from typing import List, Tuple

class HomographyReprojectionError(pyceres.CostFunction):
    def __init__(self, x1, y1, x2, y2):
        super().__init__()
        self.x2, self.y2, self.x1, self.y1 = x1, y1, x2, y2
        self.set_num_residuals(2)
        self.set_parameter_block_sizes([9, 9])  # Each homography has 9 parameters (3x3)

    def Evaluate(self, parameters, residuals, jacobians):
        H1 = np.array(parameters[0]).reshape(3, 3)
        H2 = np.array(parameters[1]).reshape(3, 3)
        p1 = np.array([self.x1, self.y1, 1.0])
        p2 = np.array([self.x2, self.y2, 1.0])
        # Apply homographies
        p1_h = H1 @ p1
        p2_h = H2 @ p2
        # Normalize
        p1_h /= p1_h[2]
        p2_h /= p2_h[2]
        # Compute reprojection error
        residuals[0] = p1_h[0] - p2_h[0]
        residuals[1] = p1_h[1] - p2_h[1]
        if jacobians is not None:
            x1, y1, x2, y2 = self.x1, self.y1, self.x2, self.y2
            def compute_jacobian(H, x, y):
                J = np.zeros((2, 9))
                denom = (H[2, 0] * x + H[2, 1] * y + H[2, 2])
                denom2 = denom * denom
                # Common terms
                u = (H[0, 0] * x + H[0, 1] * y + H[0, 2])
                v = (H[1, 0] * x + H[1, 1] * y + H[1, 2])
                # du/dh
                J[0, 0] = x / denom
                J[0, 1] = y / denom
                J[0, 2] = 1.0 / denom
                J[0, 6] = -u * x / denom2
                J[0, 7] = -u * y / denom2
                J[0, 8] = -u / denom2
                # dv/dh
                J[1, 3] = x / denom
                J[1, 4] = y / denom
                J[1, 5] = 1.0 / denom
                J[1, 6] = -v * x / denom2
                J[1, 7] = -v * y / denom2
                J[1, 8] = -v / denom2
                return J
            if jacobians[0] is not None:
                jacobians[0][:] = compute_jacobian(H1, x1, y1).flatten()
            if jacobians[1] is not None:
                jacobians[1][:] = -compute_jacobian(H2, x2, y2).flatten()
        return True

def optimize_affine_transformations_with_ceres(initial_affine_transformations, matches, num_images):
    """
    Perform global bundle adjustment using the Ceres solver to optimize the affine transformations for a collection of images.

    Args:
        initial_affine_transformations: List of 2x3 affine transformation matrices
        matches: List of tuples containing (img1_idx, img2_idx, x1, y1, x2, y2)
        num_images: Number of images

    Returns:
        List of optimized 2x3 affine transformation matrices
    """

    print("Optimizing GLOBAL affine transformations with Ceres")
    affine_params = [A.flatten().astype(np.float64) for A in initial_affine_transformations]
    problem = pyceres.Problem()

    print("Initial affine parameters:")

        # Group matches by (img1_idx, img2_idx)
    grouped_matches = defaultdict(list)

    for match in matches:
        quality_score, img1_idx, img2_idx, x1, y1, x2, y2 = match
        grouped_matches[(img1_idx, img2_idx)].append((x1, y1, x2, y2))

    # Convert to a regular dictionary
    grouped_matches_dict = dict(grouped_matches)
    for (img1_idx,img2_idx) in grouped_matches_dict:
        for (img1_idx,img2_idx) in grouped_matches_dict:
            for   x1, y1, x2, y2  in grouped_matches_dict[(img1_idx,img2_idx)]:
                cost_function = AffineReprojectionError(x1, y1, x2, y2)

                problem.add_residual_block(
                    cost_function,
                    pyceres.HuberLoss(1.0),
                    [affine_params[img1_idx], affine_params[img2_idx]]
                )

    options = pyceres.SolverOptions()
    # options.max_num_iterations = 100
    # options.linear_solver_type = pyceres.LinearSolverType.SPARSE_NORMAL_CHOLESKY
    #Andrea
    # max linear solver iterations 
    options.max_num_iterations = 100
    # options.linear_solver_type = pyceres.LinearSolverType.SPARSE_NORMAL_CHOLESKY
    options.function_tolerance = 1e-1000
    options.gradient_tolerance = 1e-4
    options.parameter_tolerance = 1e-1000
    options.linear_solver_type = pyceres.LinearSolverType.SPARSE_SCHUR
    options.use_explicit_schur_complement = False
    options.preconditioner_type = pyceres.PreconditionerType.CLUSTER_TRIDIAGONAL
    options.num_threads = 8
    options.trust_region_strategy_type = pyceres.TrustRegionStrategyType.LEVENBERG_MARQUARDT
    options.check_gradients = False
    options.gradient_check_relative_precision = 1e-6
    options.minimizer_progress_to_stdout = True
    options.function_tolerance = 1e-6


    summary = pyceres.SolverSummary()
    pyceres.solve(options, problem, summary)

    logging.info(summary.FullReport())

    # Print mean error
    final_cost = summary.final_cost
    num_residuals = summary.num_residuals
    mean_error = final_cost / num_residuals if num_residuals > 0 else 0
    print(f"Mean reprojection error: {mean_error}")

    optimized_affine_transformations = [affine_params[i].reshape(2, 3) for i in range(num_images)]

    return optimized_affine_transformations


# Optimization function using Ceres
def optimize_transformations_with_ceres(initial_transformations, matches, num_images, affine=False):
    print("Optimizing GLOBAL transformations with Ceres")
    # if affine:
    #     print("Affine")
    params = [A.flatten().astype(np.float64) for A in initial_transformations]
    problem = pyceres.Problem()

    # Convert to a regular dictionary
    grouped_matches_dict = group_matches(matches=matches)
    for (img1_idx,img2_idx) in grouped_matches_dict:
        for   x1, y1, x2, y2  in grouped_matches_dict[(img1_idx,img2_idx)]:
            if affine:
                cost_function = AffineReprojectionError(x1, y1, x2, y2)
            else:
                cost_function = HomographyReprojectionError(x1, y1, x2, y2)
            problem.add_residual_block(
                cost_function,
                pyceres.CauchyLoss(1.0) ,
                [params[img2_idx], params[img1_idx]]
            )

    options = pyceres.SolverOptions()
    options.max_num_iterations = 500
    options.function_tolerance = 1e-12
    options.gradient_tolerance = 1e-12
    options.parameter_tolerance = 1e-12
    options.use_explicit_schur_complement = True
    options.trust_region_strategy_type = pyceres.TrustRegionStrategyType.LEVENBERG_MARQUARDT
    options.linear_solver_type = pyceres.LinearSolverType.SPARSE_SCHUR

    # options.max_num_iterations = 500
    # options.function_tolerance = 1e-12
    # options.gradient_tolerance = 1e-12
    # options.parameter_tolerance = 1e-12
    # options.use_explicit_schur_complement = True
    # options.trust_region_strategy_type = pyceres.TrustRegionStrategyType.LEVENBERG_MARQUARDT
    # options.linear_solver_type = pyceres.LinearSolverType.SPARSE_SCHUR
    # options.check_gradients = True
    # options.gradient_check_relative_precision = 1e1

    options.minimizer_progress_to_stdout = True
    summary = pyceres.SolverSummary()
    pyceres.solve(options, problem, summary)
    logging.info(summary.FullReport())
    # Print mean error
    final_cost = summary.final_cost
    num_residuals = summary.num_residuals
    mean_error = final_cost / num_residuals if num_residuals > 0 else 0
    print(f"Mean reprojection error: {mean_error}")
    if affine:
        optimized_transformations = [params[i].reshape(2, 3) for i in range(num_images)]
    else:
        optimized_transformations = [params[i].reshape(3, 3) for i in range(num_images)]
    return optimized_transformations

def group_matches(matches):
    """
    Groups matches by image index pairs.
    Args:
        matches: List of tuples containing match information with each tuple in the form 
                 (quality_score, img1_idx, img2_idx, x1, y1, x2, y2).
    Returns:
        A dictionary where keys are tuples of image indices (img1_idx, img2_idx) and values are 
        lists of tuples containing corresponding point coordinates (x1, y1, x2, y2).
    """

    grouped_matches = defaultdict(list)
    for match in matches:
        quality_score, img1_idx, img2_idx, x1, y1, x2, y2 = match
        grouped_matches[(img1_idx, img2_idx)].append((x1, y1, x2, y2))
    # Convert to a regular dictionary
    return dict(grouped_matches)


def incremental_bundle_adjustment_affine_with_ceres(
    initial_affines: List[np.ndarray],
    matches: List[Tuple[int, int, int, float, float, float, float]],
    num_images: int,
    max_iterations: int = 10,
    tolerance: float = 1e-6
) -> List[np.ndarray]:
    """
    Perform iterative bundle adjustment using the Ceres solver.

    Args:
        initial_affines: List of 2x3 affine transformation matrices
        matches: List of tuples containing (img1_idx, img2_idx, x1, y1, x2, y2)
        num_images: Number of images
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance

    Returns:
        List of optimized 2x3 affine transformation matrices
    """
    affine_params = [a.flatten().astype(np.float64) for a in initial_affines]
    prev_error = float('inf')
    print("Starting Iterative Bundle Adjustment")

    # Split matches for every couple of images
    print(f"Number of matches: {len(matches)}")
    # Convert to a regular dictionary
    grouped_matches_dict = group_matches(matches=matches)
    iteration=0
    problem = pyceres.Problem()
    for (img1_idx,img2_idx) in grouped_matches_dict:
        print(f"Setting up problem for iteration {iteration}")
        for (img1_idx,img2_idx) in grouped_matches_dict:
            for  x1, y1, x2, y2 in grouped_matches_dict[(img1_idx,img2_idx)]:
                    cost_function = AffineReprojectionError(x1, y1, x2, y2)
                    problem.add_residual_block(
                        cost_function,
                        pyceres.HuberLoss(1.0),
                        [affine_params[img1_idx], affine_params[img2_idx]]  # Pass as list
                )
        print("Residual blocks added to problem")
        iteration+=1
        error_list = []  # Store errors for analysis 
        options = pyceres.SolverOptions()
        # options.max_num_iterations = 100
        # options.linear_solver_type = pyceres.LinearSolverType.SPARSE_NORMAL_CHOLESKY
        #Andrea
        # max linear solver iterations 
        options.max_num_iterations = 100
        # options.linear_solver_type = pyceres.LinearSolverType.SPARSE_NORMAL_CHOLESKY
        options.function_tolerance = 1e-1000
        options.gradient_tolerance = 1e-4
        options.parameter_tolerance = 1e-1000
        options.linear_solver_type = pyceres.LinearSolverType.SPARSE_SCHUR
        options.use_explicit_schur_complement = False
        options.preconditioner_type = pyceres.PreconditionerType.CLUSTER_TRIDIAGONAL
        options.num_threads = 8
        options.trust_region_strategy_type = pyceres.TrustRegionStrategyType.LEVENBERG_MARQUARDT
        options.check_gradients = False
        options.gradient_check_relative_precision = 1e-6
        options.minimizer_progress_to_stdout = True
        options.function_tolerance = 1e-6
        summary = pyceres.SolverSummary()
        print("Solving...")
        pyceres.solve(options, problem, summary)
        print(f"Iteration {iteration + 1} complete: Final cost = {summary.final_cost}")

        error_list.append(summary.final_cost)
        mean_error = sum(error_list) / len(error_list)
        print(f"Mean error after iteration {iteration + 1}: {mean_error}")

        prev_error = summary.final_cost

    return [affine_params[i].reshape(2, 3) for i in range(num_images)]
