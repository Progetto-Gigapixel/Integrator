import numpy as np
import cv2
from core.CeresBA import optimize_transformations_with_ceres, optimize_affine_transformations_with_ceres
from core.GridMatcher import find_center
import logging
import datetime
now = datetime.datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
logger = logging.getLogger("ImageGridStitcher")
#logging.basicConfig(filename=f'log/ImageGridStitcher-{timestamp}.log', level=logging.INFO)

class ImageGridStitcher:
    """
    A class to stitch a grid of images into a panorama using homographies.

    Usage:
        # Initialize the ImageGridStitcher with a grid of images and homographies
        stitcher = ImageGridStitcher(images_grid, homography_grid)

        # Show an image within the screen
        stitcher.show_image_within_screen(image, window_name="Image")

        # Get the size of the final panorama
        pano_width, pano_height, trans_matrix = stitcher.get_panorama_size_from_center(images_grid, homography_grid, num_rows, num_cols)

        # Get the stitched panorama image
        panorama = stitcher.get_stitched_image_from_center(save=True, image_name="merged_image.tiff", rows=num_rows, cols=num_cols, ba=True, timeLapse=True)

    Args:
        images_grid (list): A 2D list where each element is a dictionary containing image data and associated metadata.
        homography_grid (list): A 2D list where each element contains homography information for aligning images in the grid.
    """
    def __init__(self, images_grid=[], homography_grid=[] ):
        """
        Initializes the ImageGridStitcher with a grid of images and a corresponding grid of homographies.

        Args:
            images_grid (list): A 2D list where each element is a dictionary containing image data and associated metadata.
            homography_grid (list): A 2D list where each element contains homography information for aligning images in the grid.
        """

        self.images_grid= images_grid
        self.homography_grid= homography_grid
        self.global_homographies = []
        self.panorama_size = None
        return


    def get_panorama_size_from_center(self,image_grid, homography_grid, num_rows, num_cols):
        """Computes the size of the final panorama using homographies, starting from the center."""
        center_row, center_col = find_center(num_rows, num_cols)
        reference_image = image_grid[center_row][center_col]['image']
        height, width = reference_image.shape[:2]

        # Initialize global homographies with identity matrices
        global_homographies = [[np.eye(3) for _ in range(num_cols)] for _ in range(num_rows)]

        # Compute global homographies starting from the center
        # Propagate to the top-left quadrant
        for row in range(center_row, num_rows, 1):  # From center row to top
            for col in range(center_col, -1, -1):  # From center column to left
                if row > center_row:
                    global_homographies[row][col] = (
                        global_homographies[row - 1][center_col] @  homography_grid[row-1][center_col]["top"]["homography"]
                    )
                if col < center_col:
                    #print(f"Processing row: {row}, col: {col}: {homography_grid[row][col]}")
                    # Apply right homography (inverse of left homography)
                    global_homographies[row][col] = (
                        global_homographies[row][col + 1] @ homography_grid[row][col + 1 ]["left"]["homography"]
                    )

        # Propagate to the top-right quadrant
        for row in range(center_row, num_rows, 1):  # From center row to top
            for col in range(center_col, num_cols,1):  # From center column to right
                if row > center_row:
                    global_homographies[row][col] = (
                        global_homographies[row - 1][center_col] @  homography_grid[row-1][center_col]["top"]["homography"]
                    )
                if col > center_col:
                    global_homographies[row][col] = (
                        global_homographies[row][col-1] @ homography_grid[row][col-1]["right"]["homography"]
                    )

        # Propagate to the bottom-left quadrant
        for row in range(center_row, -1, -1):  # From center row to bottom
            for col in range(center_col, -1, -1):  # From center column to left
                if row < center_row:
                    global_homographies[row][col] = (
                        global_homographies[row + 1][center_col]  @ homography_grid[row + 1][center_col]["bottom"]["homography"]
                    )
                if col < center_col:
                    global_homographies[row][col] = (
                        global_homographies[row][col + 1] @ homography_grid[row][col + 1 ]["left"]["homography"]
                    )

        # Propagate to the bottom-right quadrant
        for row in range(center_row, -1, -1):  # From center row to bottom
            for col in range(center_col, num_cols,1):  # From center column to right
                if row < center_row:
                    global_homographies[row][col] = (
                        global_homographies[row + 1][center_col]  @ homography_grid[row + 1][center_col]["bottom"]["homography"]
                    )
                if col > center_col:
                    logger.info(f"Processing row: {row}, col: {col}")
                    global_homographies[row][col] = (
                        global_homographies[row][col-1] @ homography_grid[row][col-1]["right"]["homography"]
                    )


        # Compute transformed corners for all images
        image_corners = np.array(
            [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
        ).reshape(-1, 1, 2)
        transformed_corners = [
            cv2.perspectiveTransform(image_corners, global_homographies[row][col])
            for row in range(num_rows)
            for col in range(num_cols)
        ]
        transformed_corners = np.vstack(transformed_corners).reshape(-1, 2)

        # Compute panorama bounding box
        min_x, min_y = np.int32(transformed_corners.min(axis=0) - 0.5)
        max_x, max_y = np.int32(transformed_corners.max(axis=0) + 0.5)
        panorama_width = max_x - min_x
        panorama_height = max_y - min_y

        # Compute translation matrix to shift images into positive space
        translation_matrix = np.array(
            [[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float32
        )

        return panorama_width, panorama_height, translation_matrix

    def get_stitched_image_from_center(self,save=False,image_name="merged_image.tiff", rows=-1, cols=-1, ba=False, ba_num_params=100, GUI=False):
        if rows == -1:
            rows = len(self.images_grid)
        if cols == -1:
            cols = len(self.images_grid[0])
        panorama=self.merge_images_from_center(self.images_grid,self.homography_grid, rows, cols,ba=ba, ba_num_params=ba_num_params,GUI=GUI)
        if save:
            cv2.imwrite(image_name, panorama)

        return panorama

    def propagate_homographies(self, global_homographies, homography_grid,
                                num_rows, num_cols, row_range, col_range,
                                row_condition, col_condition,
                                row_offset, col_offset,
                                row_key, col_key):
        """
        Propagates homographies in both row-first and column-first order, starting from the top-left (0,0) element.
        If a homography has a large rotation angle (> 0.1 rad), it is replaced with a translation-only homography.
        The result is the average of the two propagation methods.
        """
        def get_rotation_angle(homography):
            """Extracts rotation angle in degrees from a homography matrix"""
            if homography is None:
                return 0
            try:
                # Ensure we have a proper numeric numpy array
                R = np.array(homography[:2, :2], dtype=np.float64)          
                # Check for invalid values
                if not np.all(np.isfinite(R)):
                    return 0
                # Compute SVD to get rotation angle
                u, _, vt = np.linalg.svd(R)
                R = u @ vt  # Orthogonal matrix
                # Calculate angle (in degrees)
                angle = np.arctan2(R[1, 0], R[0, 0]) * 180 / np.pi
                return angle  
            except (TypeError, ValueError, np.linalg.LinAlgError) as e:
                print(f"Warning: Could not compute rotation angle: {str(e)}")
                return 0
        
        def make_translation_only(homography):
            if homography is None:
                return None
            h = np.eye(3)
            h[0, 2] = homography[0, 2]
            h[1, 2] = homography[1, 2]
            return h

        def init_empty_grid():
            return [[None for _ in range(num_cols)] for _ in range(num_rows)]

        row_first_result = init_empty_grid()
        col_first_result = init_empty_grid()

        def propagate_row_first():
            """
            Propagates homographies in row-first order, starting from the top-left (0,0) element.
            If a homography has a large rotation angle (> 0.1 rad),
            it is replaced with a translation-only homography.
            """
            temp = [row[:] for row in global_homographies]
            for col in col_range:
                for row in row_range:
                    if col_condition(col):
                        temp[row][col] = (
                            temp[row][col + col_offset] @ 
                            homography_grid[row][col + col_offset][col_key]["homography"]
                        )
                        if get_rotation_angle(temp[row][col]) > 5:
                            print(f"[Row-first] ERROR: Large rotation at ({row},{col})")
                            temp[row][col] = make_translation_only(temp[row][col])
                        row_first_result[row][col] = temp[row][col]

            for col in col_range:
                for row in row_range:
                    if row_condition(row):
                        temp[row][col] = (
                            temp[row + row_offset][col] @ 
                            homography_grid[row + row_offset][col][row_key]["homography"]
                        )
                        if get_rotation_angle(temp[row][col]) > 5:
                            print(f"[Row-first] ERROR: Large rotation at ({row},{col})")
                            temp[row][col] = make_translation_only(temp[row][col])
                        row_first_result[row][col] = temp[row][col]

        def propagate_column_first():
            """
            Propagates homographies in column-first order, starting from the top-left (0,0) element.
            If a homography has a large rotation angle (> 0.1 rad), it is replaced with a translation-only homography.
            The result is stored in col_first_result.
            """
            temp = [row[:] for row in global_homographies]
            for row in row_range:
                for col in col_range:
                    if row_condition(row):
                        temp[row][col] = (
                            temp[row + row_offset][col] @ 
                            homography_grid[row + row_offset][col][row_key]["homography"]
                        )
                        if get_rotation_angle(temp[row][col]) > 5:
                            print(f"[Col-first] ERROR: Large rotation at ({row},{col})")
                            temp[row][col] = make_translation_only(temp[row][col])
                        col_first_result[row][col] = temp[row][col]

            for row in row_range:
                for col in col_range:
                    if col_condition(col):
                        temp[row][col] = (
                            temp[row][col + col_offset] @ 
                            homography_grid[row][col + col_offset][col_key]["homography"]
                        )
                        if get_rotation_angle(temp[row][col]) > 5:
                            print(f"[Col-first] ERROR: Large rotation at ({row},{col})")
                            temp[row][col] = make_translation_only(temp[row][col])
                        col_first_result[row][col] = temp[row][col]

        propagate_column_first()
        propagate_row_first()

        # Compute average of row_first_result and col_first_result
        averaged_result = init_empty_grid()
        for row in range(num_rows):
            for col in range(num_cols):
                h1 = row_first_result[row][col]
                h2 = col_first_result[row][col]
                if h1 is not None and h2 is not None:
                    averaged_result[row][col] = (h1 + h2) / 2
                    # print(h1-h2)
                    # print(f"{row}:{col} ha 2 path e faccio la media")
                elif h1 is not None:
                    averaged_result[row][col] = h1
                    # print(f"{row}:{col} ha 1 path row first")
                elif h2 is not None:
                    averaged_result[row][col] = h2
                    # print(f"{row}:{col} ha 1 path col first")
                else:
                    averaged_result[row][col] = None
        return averaged_result

    def merge_images_from_center(self, image_grid, homography_grid, rows, cols, ba=False,ba_num_params=100, GUI=False):
        """
        Merges images from the center using homographies propagated from the center.

        Args:
            image_grid (list): A 2D list of dictionaries containing image data and associated metadata.
            homography_grid (list): A 2D list of dictionaries containing homography information for aligning images in the grid.
            rows (int): Number of rows in the grid.
            cols (int): Number of columns in the grid.
            timeLapse (bool, optional): If True, displays the panorama as it is being built. Defaults to False.
            ba (bool, optional): If True, performs bundle adjustment on the homographies. Defaults to False.

        Returns:
            numpy.ndarray: The merged panorama image.
        """
        num_rows= rows
        num_cols= cols
        logger.info("Stitching image using center-based homographies")
       # Compute global homographies for all images
        global_homographies = [[np.eye(3) for _ in range(cols)] for _ in range(rows)]
        # Center of the grid (fix problem of small grid 2x2)
        center_row, center_col = find_center(num_rows, num_cols)
        #Initialize global homographies with identity matrices
        global_homographies_ini = [[np.eye(3) for _ in range(num_cols)] for _ in range(num_rows)]
        num_directions = self.get_directions(num_rows, num_cols, center_row, center_col)
        # direction --> 
        #print(f"center {center_row}:{center_col}")
        res = [[None for _ in range(num_cols)] for _ in range(num_rows)]  # Initialize 2D result grid
        for args in num_directions:
            partial_res = self.propagate_homographies(
                global_homographies_ini, homography_grid,  num_rows, num_cols, *args
            )
            if partial_res is not None:
                for i in range(num_rows):
                    for j in range(num_cols):
                        if partial_res[i][j] is not None:  # Override only where partial_res has a value
                            res[i][j] = partial_res[i][j]
        res[center_row][center_col]=np.eye(3)
        global_homographies=res
        if ba:
            logger.info("converting to affine trasformations")
            # extract upto 100 matches extend to 8 neighbors
            print(f"ba_num_params {ba_num_params}")
            if GUI:
                matches = self.exact_matches_GUI(homography_grid, rows, cols, ba_num_params)
            else:
                matches = self.exact_matches(homography_grid, rows, cols, ba_num_params)
            # Convert homographies to affine transformations (extract first two rows)
            initial_affines = [H[:2, :] for row in global_homographies for H in row]
            logger.info("Incremental adjustment")
            #optimized_affines= incremental_bundle_adjustment_affine_with_ceres(initial_affines, matches, rows * cols)
            # Optimize affine transformations (GLOBAL)
            logger.info("Final global adjustment")
            optimized_affines = optimize_transformations_with_ceres(initial_affines, matches, rows * cols,True)
            # Convert optimized affine transformations back to homographies
            optimized_homographies = [np.eye(3) for _ in range(rows * cols)]
            for row in range(rows):
                for col in range(cols):
                    optimized_homographies[row * cols + col][:2, :] = optimized_affines[row * cols + col]
            logger.info("converting back to homographies")
            optimized_homography_grid = np.array(optimized_homographies).reshape(rows, cols, 3, 3)
            global_homographies =  optimized_homography_grid
            logger.info("Final global adjustment done")
        logger.info("Calculating final panorama size")
        pano_width, pano_height, trans_matrix = self.get_panorama_size_from_center(
            image_grid, homography_grid, rows, cols
        )

        for row in range(num_rows):
            for col in range(num_cols):
                global_homographies[row][col] = trans_matrix @ global_homographies[row][col]
        # save global homographies for later usage
        self.global_homographies = global_homographies
        self.panorama_size = (pano_height, pano_width)
        panorama=self.stitch_images(image_grid)
        return panorama


    def stitch_images(self, image_grid):
        traversal_order = self.spiral_traversal_from_center(len(image_grid),len(image_grid[0]))
        pano_height, pano_width = self.panorama_size
        panorama = np.zeros((pano_height, pano_width), dtype=np.uint8)

        for row, col in traversal_order:
            full_homography = self.global_homographies[row][col]
            logger.info(f"Transforming image at ({row},{col}) x:{full_homography[0][2]}, y:{full_homography[1][2]}")
            warped_img = cv2.warpPerspective(
                image_grid[row][col]["image"], full_homography, (pano_width,pano_height)
            )

            blend_mask = (warped_img > 0).astype(np.uint8)
            panorama[blend_mask == 1] = warped_img[blend_mask == 1]
        return panorama
    def stitch_images_rgb(self, image_grid):
        """
        Stitch RGB images together using precomputed homographies

        Args:
            image_grid: 2D grid of images with their metadata

        Returns:
            stitched RGB panorama image
        """
        #extract row and cols from image grid
        rows = len(self.images_grid)
        cols = len(self.images_grid[0])
        pano_width, pano_height, tra_matrix=self.get_panorama_size_from_center(self.images_grid, self.homography_grid,rows,cols)
        traversal_order = self.spiral_traversal_from_center(len(image_grid), len(image_grid[0]))
        self.panorama_size= (pano_height, pano_width)

        # Initialize panorama as black RGB image
        panorama = np.zeros((pano_height, pano_width, 3), dtype=np.uint8)

        for row, col in traversal_order:
            full_homography = self.global_homographies[row][col]
            logger.info(f"Transforming image at ({row},{col}) x:{full_homography[0][2]}, y:{full_homography[1][2]}")

            # Warp RGB image
            warped_img = cv2.warpPerspective(
                image_grid[row][col]["image"],
                full_homography,
                (pano_width, pano_height),
                flags=cv2.INTER_LINEAR
            ) 
            # Create blend mask (checking all 3 channels)
            if len(warped_img.shape) == 3:  # Ensure the image is RGB
                blend_mask = (warped_img > 0).all(axis=-1).astype(np.uint8)

                # Blend using mask (for all 3 channels)
                for c in range(3):  # R, G, B channels
                    panorama[:, :, c][blend_mask == 1] = warped_img[:, :, c][blend_mask == 1]
            else:
                logger.warning(f"Image at ({row},{col}) is not an RGB image.")
        return panorama

    def get_directions(self, num_rows, num_cols, center_row, center_col):
        """
        Returns a list of directions for propagating homographies from the center image.

        The directions are represented as a tuple of:
            - A range of row indices
            - A range of column indices
            - A function that takes a row index and returns True if the row is in the range
            - A function that takes a column index and returns True if the column is in the range
            - An offset for the row index
            - An offset for the column index
            - A string indicating the direction of propagation (top, bottom, left, right)

        The directions are ordered such that the first direction is the top-left quadrant, the second direction is the top-right quadrant, the third direction is the bottom-left quadrant, and the fourth direction is the bottom-right quadrant.
        """
        num_directions = [
            (range(center_row, num_rows, 1), range(center_col, -1, -1), lambda r: r > center_row, lambda c: c < center_col, -1, 1, "top", "left"),
            (range(center_row, num_rows, 1), range(center_col, num_cols, 1), lambda r: r > center_row, lambda c: c > center_col, -1, -1, "top", "right"),
            (range(center_row, -1, -1), range(center_col, -1, -1), lambda r: r < center_row, lambda c: c < center_col, 1, 1, "bottom", "left"),
            (range(center_row, -1, -1), range(center_col, num_cols, 1), lambda r: r < center_row, lambda c: c > center_col, 1, -1, "bottom", "right")
        ]

        return num_directions

    def exact_matches_GUI(self, homography_grid, rows, cols, max_matches_number=250):
        all_matches = []
        directions = {
            'right': (0, 1),
            'left': (0, -1),
            'top': (1, 0),
            'bottom': (-1, 0),
            'top_right': (1, 1),
            'top_left': (1, -1),
            'bottom_right': (-1, 1),
            'bottom_left': (-1, -1)
        }
        # Get traversal order
        #traversal_order = self.radial_traversal_from_center(rows, cols)
        traversal_order = self.spiral_traversal_from_center(rows, cols)

        for (i, j) in traversal_order:
            if homography_grid[i][j] is None:
                continue

            for dir_name, (di, dj) in directions.items():
                # Skip if this direction doesn't exist in the grid
                if dir_name not in homography_grid[i][j] or homography_grid[i][j][dir_name] is None:
                    continue
                # Calculate neighbor coordinates
                ni, nj = i + di, j + dj
                if ni < 0 or ni >= rows or nj < 0 or nj >= cols:
                    continue
                # Get match data
                match_data = homography_grid[i][j][dir_name]
                src_pts = match_data['src_pts']
                dst_pts = match_data['dst_pts']
                inliers_mask = match_data['inliers_mask']

                # Get inlier points
                dst_pts_inliers = dst_pts[inliers_mask]
                src_pts_inliers = src_pts[inliers_mask]

                # compute the quality score of a match
                quality_score = np.sum(inliers_mask)

                current_idx = i * cols + j
                neighbor_idx = ni * cols + nj
                for (x1, y1), (x2, y2) in zip(src_pts_inliers,dst_pts_inliers ):
                    all_matches.append((quality_score, current_idx, neighbor_idx, x1, y1, x2, y2))

        all_matches.sort(reverse=True, key=lambda x: x[0])

        pair_counts = {}
        filtered_matches = []

        for match in all_matches:
            quality_score, current_idx, neighbor_idx, x1, y1, x2, y2 = match
            pair_key = (current_idx, neighbor_idx)
            # Get current count for this pair (default to 0 if not found)
            count = pair_counts.get(pair_key, 0)   
            # If we haven't reached 250 for this pair, add to filtered matches
            if count < 5:
                filtered_matches.append(match)
                pair_counts[pair_key] = count + 1

        # Replace all_matches with the filtered version
        all_matches = filtered_matches

        return all_matches
    
    def exact_matches(self, homography_grid, rows, cols, max_matches_number=100):
        all_matches = []
        directions = {
            'right': (0, 1),
            'left': (0, -1),
            'top': (1, 0),
            'bottom': (-1, 0),
            'top_right': (1, 1),
            'top_left': (1, -1),
            'bottom_right': (-1, 1),
            'bottom_left': (-1, -1)
        }
        # Get traversal order
        #traversal_order = self.radial_traversal_from_center(rows, cols)
        traversal_order = self.spiral_traversal_from_center(rows, cols)

        for (i, j) in traversal_order:
            if homography_grid[i][j] is None:
                continue

            for dir_name, (di, dj) in directions.items():
                # Skip if this direction doesn't exist in the grid
                if dir_name not in homography_grid[i][j] or homography_grid[i][j][dir_name] is None:
                    continue
                # Calculate neighbor coordinates
                ni, nj = i + di, j + dj
                if ni < 0 or ni >= rows or nj < 0 or nj >= cols:
                    continue
                # Get match data
                match_data = homography_grid[i][j][dir_name]
                src_pts = match_data['src_pts']
                dst_pts = match_data['dst_pts']
                inliers_mask = match_data['inliers_mask']
                # compute the quality score of a match
                quality_score = match_data['scores']
                # #### LOOK CAREFULLY HERE ####
                # quality_score = np.ones_like(inliers_mask, dtype=float)  # Create an array of ones with the same shape as 
                # ####
                # Get inlier points
                dst_pts_inliers = dst_pts[inliers_mask]
                src_pts_inliers = src_pts[inliers_mask]
                inlier_scores = np.array(quality_score)[inliers_mask]

                current_idx = i * cols + j
                neighbor_idx = ni * cols + nj
                for (x1, y1), (x2, y2), quality_score in zip(src_pts_inliers,dst_pts_inliers,inlier_scores ):
                    all_matches.append((quality_score, current_idx, neighbor_idx, x1, y1, x2, y2))

        all_matches.sort(reverse=False, key=lambda x: x[0])

        pair_counts = {}
        filtered_matches = []

        for match in all_matches:
            quality_score, current_idx, neighbor_idx, x1, y1, x2, y2 = match
            pair_key = (current_idx, neighbor_idx)
            # Get current count for this pair (default to 0 if not found)
            count = pair_counts.get(pair_key, 0)   
            # If we haven't reached 250 for this pair, add to filtered matches
            if count < max_matches_number:
                filtered_matches.append(match)
                pair_counts[pair_key] = count + 1

        # Replace all_matches with the filtered version
        all_matches = filtered_matches

        return all_matches

    def radial_traversal_from_center(self,rows, cols):
        """
        Generates a radial traversal order starting from the center of the grid.
        :param rows: Number of rows in the grid.
        :param cols: Number of columns in the grid.
        :return: A list of (row, col) indices in radial order.
        """
        center_row, center_col = find_center(rows,cols)
        traversal_order = []
        # Define the maximum radius (distance from the center to the farthest corner)
        max_radius = max(center_row, center_col, rows - center_row - 1, cols - center_col - 1)
        for radius in range(max_radius + 1):
            # Iterate over the perimeter of the current radius
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if abs(dr) == radius or abs(dc) == radius:  # Only process the perimeter
                        row = center_row + dr
                        col = center_col + dc
                        if 0 <= row < rows and 0 <= col < cols:  # Ensure indices are within bounds
                            traversal_order.append((row, col))

        return traversal_order
    def spiral_traversal_from_center(self, rows, cols):
        """
        Generates a spiral traversal order starting from the center of the grid.

        :param rows: Number of rows in the grid.
        :param cols: Number of columns in the grid.
        :return: A list of (row, col) indices in spiral order.
        """
        center_row, center_col = find_center(rows, cols)
        traversal_order = [(center_row, center_col)]

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
        step_size = 1  # Number of steps to take in a given direction
        row, col = center_row, center_col  # Start from the center

        while len(traversal_order) < rows * cols:
            for d in range(4):  # Loop through directions
                dr, dc = directions[d]
                for _ in range(step_size):  # Move in the current direction
                    row += dr
                    col += dc
                    if 0 <= row < rows and 0 <= col < cols:  # Ensure within bounds
                        traversal_order.append((row, col))
                    if len(traversal_order) == rows * cols:
                        return traversal_order  # Stop early if we covered the grid

                if d % 2 == 1:  # Increase step size after every two directions (right, down)
                    step_size += 1

        return traversal_order

    def getGlobalHomographies(self):
        return self.global_homographies
    def getPanoramaSize(self):
        return self.panorama_size
    
    def setGlobalHomography(self, gh):
        self.global_homographies = gh

    