import cv2
from cv2.typing import IndexParams, SearchParams
import numpy as np
import logging
import datetime
now = datetime.datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
logger = logging.getLogger("Grid Matcher")
#logging.basicConfig(filename=f'Grid-Matcher-{timestamp}.log', level=logging.INFO)
import time

RANSAC_THRESHOLD = 1.0
MATCH_DISTANCE = 0.7
MIN_INLIERS = 16
DEBUG = True
FLANN_INDEX_KDTREE = 1

def get_flann():
    # Initialize FLANN-based matcher
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    return flann

def match_adjacent_images_from_center_extended(images_grid, rows, cols, match_distance=MATCH_DISTANCE, min_inliers=MIN_INLIERS, ransac_threshold=RANSAC_THRESHOLD):
    """
    Matches adjacent images in a grid starting from the center, extending to all eight neighboring directions.

    Args:
        images_grid (list): A 2D list where each element is a dictionary containing image data and associated metadata.
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.
        match_distance (float, optional): The distance ratio for filtering good matches. Defaults to 0.7.
        min_inliers (int, optional): Minimum number of inliers required to consider a match valid. Defaults to 4.
        ransac_threshold (float, optional): The RANSAC reprojection threshold. Defaults to 1.0.
        debug (bool, optional): If True, enables debug logging. Defaults to False.

    Returns:
        list: A 2D list of dictionaries containing match data for each image in the grid, with keys for each of the eight possible neighbor directions.
    """
    print(f"------match_distance {match_distance}")
    print(f"------ransac {ransac_threshold}")
    print(f"------min_inliers {min_inliers}")

    overall_start = time.time()
    # Initialize a list to store matches
    matches_list = [[None for _ in range(cols)] for _ in range(rows)]
    # Find the center of the grid
    # center_row, center_col = find_center(rows, cols)


    # Cardinal directions (original matches)direction_map = {
    direction_map = {
        (0, 1): "right",
        (0, -1): "left",
        (1, 0): "top",
        (-1, 0): "bottom",
        (-1, -1): "bottom_left",
        (-1, 1): "bottom_right",
        (1, -1): "top_left",
        (1, 1): "top_right"
    }
    # Iterate over the grid starting from the center
    total_numbers = rows * cols
    matching_progress = 0
    matching_progress_step = 100 / total_numbers
    for i in range(rows):
        for j in range(cols):
            if images_grid[i][j] is None:
                continue
            match_data = {
                'right': None,      # Homography and points for the right neighbor
                'left': None,       # Homography and points for the left neighbor
                'top': None,        # Homography and points for the top neighbor
                'bottom': None,    # Homography and points for the bottom neighbor
                'top_left': None,   # Homography and points for the top-left neighbor
                'top_right': None, # Homography and points for the top-right neighbor
                'bottom_left': None, # Homography and points for the bottom-left neighbor
                'bottom_right': None # Homography and points for the bottom-right neighbor
            }
            for delta_row, delta_col in direction_map:
                neighbor_row = i + delta_row
                neighbor_col = j + delta_col
                direction = direction_map[(delta_row, delta_col)]

                if 0 <= neighbor_row < rows and 0 <= neighbor_col < cols:
                    match_data[direction] = process_match(images_grid, i, j, neighbor_row, neighbor_col, direction, ransac_threshold,match_distance, min_inliers)
            matches_list[i][j] = match_data
            matching_progress = matching_progress + matching_progress_step
            print(f"Matching progress:  ({(matching_progress):.2f}%)",flush=True)
    
    overall_end = time.time()
    print(f"Total execution time: {overall_end - overall_start:.2f} seconds")
    return matches_list


def find_center(rows, cols):
    """
    Finds the center of a matrix, taking into account even dimensions.

    Args:
        rows (int): Number of rows in the matrix.
        cols (int): Number of columns in the matrix.

    Returns:
        tuple: A tuple containing the center row and center column.
    """
    center_row = (rows - 1) // 2 if rows % 2 == 0 else rows // 2
    center_col = (cols - 1) // 2 if cols % 2 == 0 else cols // 2
    return center_row, center_col

def compute_match_scores(good_matches):
    scores = []
    for m in good_matches:
        # Score = 1 / distance (add small epsilon to avoid division by zero)
        score = 1.0 / (m.distance + 1e-6)
        scores.append(score)
    return scores
def in_overlap_area(kp, w, h, direction, is_src):
    x, y = kp
    # Source vs Destination side filtering
    if direction == "right":
        return x >= w / 2 if is_src else x < w / 2
    elif direction == "left":
        return x < w / 2 if is_src else x >= w / 2
    elif direction == "bottom":
        return y >= h / 2 if is_src else y < h / 2
    elif direction == "top":
        return y < h / 2 if is_src else y >= h / 2
    elif direction == "bottom_right":
        return (x >= w / 2 and y >= h / 2) if is_src else (x < w / 2 and y < h / 2)
    elif direction == "bottom_left":
        return (x < w / 2 and y >= h / 2) if is_src else (x >= w / 2 and y < h / 2)
    elif direction == "top_right":
        return (x >= w / 2 and y < h / 2) if is_src else (x < w / 2 and y >= h / 2)
    elif direction == "top_left":
        return (x < w / 2 and y < h / 2) if is_src else (x >= w / 2 and y >= h / 2)
    return True  # fallback (shouldn't occur)

def process_match(images_grid, i1, j1, i2, j2, direction, ransac_threshold, match_distance, min_inliers):
    if images_grid[i2][j2] is None:
        return None
    if DEBUG:
        logger.debug(f"img {i1},{j1} -> {direction}")
    kps1 = images_grid[i1][j1]['keypoints']
    kps2 = images_grid[i2][j2]['keypoints']
    desc1 = images_grid[i1][j1]['descriptors']
    desc2 = images_grid[i2][j2]['descriptors']
    img_shape = images_grid[i1][j1]['image'].shape
    if desc1 is None or desc2 is None:
        return None
    h, w = img_shape[:2]
    # Filter keypoints/descriptors in the overlapping area
    mask1 = [idx for idx, kp in enumerate(kps1) if in_overlap_area(kp, w, h, direction, is_src=True)]
    mask2 = [idx for idx, kp in enumerate(kps2) if in_overlap_area(kp, w, h, direction, is_src=False)]
    if len(mask1) < 2 or len(mask2) < 2:
        return None
    kps1_filt = [kps1[i] for i in mask1]
    desc1_filt = np.array([desc1[i] for i in mask1])
    kps2_filt = [kps2[i] for i in mask2]
    desc2_filt = np.array([desc2[i] for i in mask2])
    
    flann = get_flann()
    matches = flann.knnMatch(desc1_filt, desc2_filt, k=2)
    good_matches = [m for m, n in matches if m.distance < match_distance * n.distance]

    if len(good_matches) < min_inliers:
        if DEBUG:
            print(f"Too few good matches: {len(good_matches)}")
        return None

    src_pts = np.float32([kps1_filt[m.queryIdx] for m in good_matches])
    dst_pts = np.float32([kps2_filt[m.trainIdx] for m in good_matches])

    while True:
        A, inliers = cv2.estimateAffinePartial2D(from_=dst_pts, to=src_pts,
                                                method=cv2.RANSAC,
                                                ransacReprojThreshold=ransac_threshold)
        inliers_mask = inliers.ravel().astype(bool)
        if inliers is not None and np.count_nonzero(inliers_mask)> MIN_INLIERS:
            break
        ransac_threshold = ransac_threshold*2
        if ransac_threshold > 100:
            return None
        #print(f"Not enough inliers after RANSAC provi ad aumentare: {ransac_threshold}")
    if A is None or inliers is None:
        return None


    inliers_mask = inliers.ravel().astype(bool)

    if np.count_nonzero(inliers_mask) < MIN_INLIERS:
        if DEBUG:
            print(f"Not enough inliers after RANSAC: {np.count_nonzero(inliers_mask)}")
        return None

    H = np.array([
        [A[0, 0], A[0, 1], A[0, 2]],
        [A[1, 0], A[1, 1], A[1, 2]],
        [0, 0, 1]
    ])

    if DEBUG:
        print(f"{i1},{j1} -> {direction}: {len(matches)} raw, {len(good_matches)} good, {np.count_nonzero(inliers_mask)} inliers , {ransac_threshold}")

    scores = compute_match_scores(good_matches)

    return {
        'src_pts': src_pts,
        'dst_pts': dst_pts,
        'homography': H,
        'inliers_mask': inliers_mask,
        'scores': scores
    }
