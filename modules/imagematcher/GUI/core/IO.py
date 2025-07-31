import os
import re
import cv2
import numpy as np
import h5py
import logging
import datetime
from core.WallisFilter import WallisFilter
now = datetime.datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
logger = logging.getLogger("IO")
#logging.basicConfig(filename=f'log/IO-{timestamp}.log', level=logging.INFO)
def extract_folder_indices(folder_name):
    # Use regex to match the pattern: one uppercase letter followed by one or more digits
    match = re.match(r"^([A-Z])(\d+)$", folder_name)
    if not match:
        raise ValueError(f"Folder name '{folder_name}' does not match expected pattern: one uppercase letter followed by digits")
    # Extract the letter and number
    letter = match.group(1)  # The uppercase letter (e.g., 'A', 'B')
    number = match.group(2)  # The number (e.g., '1', '2')
    # Convert the letter to a row index (A=0, B=1, ..., Z=25)
    row_index = ord(letter.upper()) - ord('A')
    # Convert the number to a column index (subtract 1 to make it zero-based)
    col_index = int(number) - 1
    return row_index, col_index


def loadImagesFromFolder(base_folder : str, type : str = "albedo", debug : bool = False, loadSIFT: bool= True, siftoptions : dict = {}, only_images=False, wallis : bool = True):
    image_extensions = [".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp", ".gif"]
    # type can be albedo, normals and reflectionMap
    # load all subfolders
    subfolders =[f.path for f in os.scandir(base_folder) if f.is_dir() and  re.match(r"^([A-Z])(\d+)$",os.path.basename(f.path))]
    if debug: logger.info(f'found {len(subfolders)} subfolders')
    #Wallis 100 filter
    wallisFilter=WallisFilter()
    # initiate the grid dimension
    max_row = 0
    max_col = 0
    for folder_path in subfolders:
        folder_name = os.path.basename(folder_path)
        row_index, col_index = extract_folder_indices(folder_name)
        max_row = max(max_row, row_index)
        max_col = max(max_col, col_index)

    images_grid = [[{} for _ in range(max_col + 1)] for _ in range(max_row + 1)]
    for subfolder in subfolders:
        folder_name = os.path.basename(subfolder)
        file_name= ""
        row_index, col_index = extract_folder_indices(folder_name)
        if debug: logger.info(f'--- folder {folder_name} --> row {row_index} col {col_index}')
        file_path = ""
        # Iterate through all files in the folder
        for fname in os.listdir(subfolder):
            # Check if the file ends with the specified type and has a .tif extension
            if any(fname.endswith(f"{type}{ext}") for ext in image_extensions):
                file_path=os.path.join(subfolder, fname)
                file_name = fname
                break
        if file_path == "":
            logger.info(f'not found any {type} image in {subfolder}')
            continue
        if debug: logger.info(file_path)
        if loadSIFT:
            sift = cv2.SIFT_create(**siftoptions)
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                logger.warning(f"Warning: Unable to read image {file_path}. Skipping.")
                return {}
            if wallis and type == 'albedo':
                #print("applying wallis filter")
                image = wallisFilter.apply(image)
            # Detect SIFT keypoints and descriptors
            keypoints, descriptors = sift.detectAndCompute(image, None)  # <- sift.detectAndCompute(image, None) None per cosa sta
            # Convert keypoints to a list of (x, y) coordinates
            keypoints_list = [kp.pt for kp in keypoints]  # List of (x, y) tuples
            image_object = {
                'name': file_name,  # Name of the image file
                'path': file_path,  # Full path to the image
                'folder': folder_name,
                'image': image,  # Numpy array of the image
                'keypoints': keypoints_list,  # List of (x, y) keypoint coordinates
                'descriptors': descriptors  # SIFT descriptors (numpy array)
            }
            images_grid[row_index][col_index] = image_object
        else:
            images_grid[row_index][col_index] = {'path': file_path }
            if only_images:
                # read in rgb
                images_grid[row_index][col_index]['image'] = cv2.imread(file_path, cv2.IMREAD_COLOR)
    return images_grid
def load_grid_names(base_folder : str):
    subfolders =[f.path for f in os.scandir(base_folder) if f.is_dir() and  re.match(r"^([A-Z])(\d+)$",os.path.basename(f.path))]

    # initiate the grid dimension
    max_row = 0
    max_col = 0
    for folder_path in subfolders:
        folder_name = os.path.basename(folder_path)
        row_index, col_index = extract_folder_indices(folder_name)
        max_row = max(max_row, row_index)
        max_col = max(max_col, col_index)

    folder_names = [[{} for _ in range(max_col + 1)] for _ in range(max_row + 1)]
    for subfolder in subfolders:
        folder_name = os.path.basename(subfolder)
        row_index, col_index = extract_folder_indices(folder_name)
        folder_names[row_index][col_index] =  folder_name
    return folder_names
def save_images_grid_to_file(save_path: str, images_grid: list[list[dict[str, any]]]):
    """Save the images grid structure to an HDF5 file."""
    with h5py.File(save_path, 'w') as hf:
        grid_shape = (len(images_grid), len(images_grid[0]))  # Store shape

        # Create dataset for shape
        hf.attrs['grid_shape'] = grid_shape

        for row_idx, row in enumerate(images_grid):
            for col_idx, img_data in enumerate(row):
                if not img_data:  # Skip empty cells
                    continue

                # Create a group for each image
                group_name = f"{row_idx}_{col_idx}"
                grp = hf.create_group(group_name)

                # Store string data
                grp.attrs['name'] = img_data['name']
                grp.attrs['path'] = img_data['path']

                # Convert keypoints (list of (x,y) tuples) to NumPy array
                keypoints_array = np.array(img_data['keypoints'], dtype=np.float32)
                grp.create_dataset('keypoints', data=keypoints_array)

                # Store descriptors (variable-sized array)
                grp.create_dataset('descriptors', data=img_data['descriptors'], dtype=np.float32)

def restore_images_grid(save_path : str) -> dict:
    """Load the images grid structure from an HDF5 file."""
    images_grid = []
    with h5py.File(save_path, 'r') as hf:
        grid_shape = hf.attrs['grid_shape']
        # Initialize an empty grid
        images_grid = [[{} for _ in range(grid_shape[1])] for _ in range(grid_shape[0])]

        for group_name in hf.keys():
            row_idx, col_idx = map(int, group_name.split("_"))
            grp = hf[group_name]

            # Extract stored attributes
            name = grp.attrs['name']
            path = grp.attrs['path']

            # Load keypoints (convert back to list of tuples)
            keypoints_array = grp['keypoints'][:]
            keypoints = [tuple(kp) for kp in keypoints_array]

            # Load descriptors
            descriptors = grp['descriptors'][:]

            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            # Reconstruct dictionary
            images_grid[row_idx][col_idx] = {
                'name': name,
                'path': path,
                'image': image,
                'keypoints': keypoints,
                'descriptors': descriptors
            }


    return images_grid



def save_matches_to_h5(file_path, matches_list):
    """Save matches_list to an HDF5 file."""
    grid_shape = (len(matches_list), len(matches_list[0]))
    with h5py.File(file_path, 'w') as f:
        f.attrs['grid_shape'] = grid_shape
        for i, row in enumerate(matches_list):
            for j, match_data in enumerate(row):
                if match_data is None:
                    continue
                group = f.create_group(f"match_{i}_{j}")

                for direction in ['right', 'left', 'top', 'bottom', 'top_right', 'top_left', 'bottom_right', 'bottom_left']:
                    if match_data[direction] is not None:
                        d_group = group.create_group(direction)
                        d_group.create_dataset("src_pts", data=np.array(match_data[direction]['src_pts'], dtype=np.float32))
                        d_group.create_dataset("dst_pts", data=np.array(match_data[direction]['dst_pts'], dtype=np.float32))
                        d_group.create_dataset("homography", data=np.array(match_data[direction]['homography'], dtype=np.float32))
                        d_group.create_dataset("inliers_mask", data=np.array(match_data[direction]['inliers_mask'], dtype=np.bool))
                        d_group.create_dataset("scores", data=np.array(match_data[direction]['scores'], dtype=np.bool))

def load_matches_from_h5(file_path) -> list[list[object]]:
    """Load matches_list from an HDF5 file."""
    with h5py.File(file_path, 'r') as f:
        grid_shape = f.attrs['grid_shape']
        rows,cols = grid_shape
        matches_list = [[None for _ in range(cols)] for _ in range(rows)]
        for key in f.keys():
            i, j = map(int, key.split('_')[1:])
            match_data = {'right': None, 'left': None, 'top': None, 'bottom': None, 'top_right': None, 'top_left': None, 'bottom_right': None, 'bottom_left': None}

            group = f[key]
            for direction in group.keys():
                d_group = group[direction]
                match_data[direction] = {
                    'src_pts': d_group["src_pts"][:],
                    'dst_pts': d_group["dst_pts"][:],
                    'homography': d_group["homography"][:],
                    'inliers_mask': d_group["inliers_mask"][:],
                    'scores': d_group["scores"][:]
                }
            matches_list[i][j] = match_data

    return matches_list

def save_global_homography_to_h5(file_path, global_homography,final_pano_size):
    """Save matches_list to an HDF5 file."""
    grid_shape = (len(global_homography), len(global_homography[0]))
    # gloval homography is grid of homography
    with h5py.File(file_path, 'w') as f:
        f.attrs['grid_shape'] = grid_shape
        f.attrs['final_pano_size'] = final_pano_size
        for i, row in enumerate(global_homography):
            for j, match_data in enumerate(row):
                if match_data is None:
                    continue
                group = f.create_group(f"match_{i}_{j}")
                # no direction
                group.create_dataset("homography", data=np.array(match_data, dtype=np.float32))

def save_global_homography_to_txt(file_path, global_homography, pano_size):
    """Save global_homography to a human-readable text file."""
    with open(file_path, 'w') as f:
        (pano_x, pano_y) = pano_size
        f.write(f"Grid shape: {len(global_homography)} x {len(global_homography[0])}\n\n")
        f.write(f"Final pano size: {pano_x}:{pano_y}\n\n")
        for i, row in enumerate(global_homography):
            for j, homography in enumerate(row):
                f.write(f"Homography at position ({i}, {j}):\n")
                if homography is None:
                    f.write("None\n\n")
                else:
                    # Convert to numpy array if it isn't already
                    homography_arr = np.array(homography)
                    # Write each row of the homography matrix
                    for row_idx in range(homography_arr.shape[0]):
                        row_str = " ".join([f"{val:.6f}" for val in homography_arr[row_idx]])
                        f.write(row_str + "\n")
                    f.write("\n")  # Add extra newline between matrices
def load_global_homo_from_h5(file_path):
    """Load matches_list from an HDF5 file."""
    with h5py.File(file_path, 'r') as f:
        grid_shape = f.attrs['grid_shape']
        rows,cols = grid_shape
        final_pano_size=f.attrs['final_pano_size']
        global_homography = [[None for _ in range(cols)] for _ in range(rows)]
        for key in f.keys():
            i, j = map(int, key.split('_')[1:])
            global_homography[i][j] = f[key]["homography"][:]
    return global_homography, final_pano_size

def save_homographies_as_cam(
    homographies,
    image_names,image_grid,pano_size,
    output_dir="texture",
    f=27457.45650249844, d0=0.0, d1=0.0,
    paspect=1.0, ppx=0.5, ppy=0.5, width=11656, height=8742, image_dir="", fnormalize=2.355650009
):
    """
    Save 4x4 homography matrices to .CAM files using image names.
    homographies and image_names are both 2D arrays (same shape).
     cx=5828 cy=4371
     -cx - dx, -cy - dy
    """ 
    
    fnormalize = 2.355650009
    f=fnormalize*width
    for i in range(len(image_names)):
            print(homographies[i])
            matrix = homographies[i]
            img_name = image_names[i]
            print(f"Processing image: {img_name}")
            base_name = os.path.splitext(os.path.basename(img_name))[0]
            ext_output_dir= os.path.join(image_dir,output_dir)
            if not os.path.exists(ext_output_dir):
                os.makedirs(ext_output_dir)
            cam_path = os.path.join(ext_output_dir, f"{base_name}.cam")
            pano_width, pano_height = pano_size
            H = np.array(matrix)
            warped_img = cv2.warpPerspective(
                image_grid[i]["image"],
                H,
                (pano_width, pano_height),
                flags=cv2.INTER_LINEAR
            )
            
            # Create blend mask (checking all 3 channels)
            if len(warped_img.shape) == 3:  # Ensure the image is RGB
                blend_mask = (warped_img > 0).all(axis=-1).astype(np.uint8)
                                                # Find the bounding rectangle of the non-zero region
                coords = cv2.findNonZero(blend_mask)
                if coords is not None:  # Only proceed if there are non-zero pixels
                    x, y, w, h = cv2.boundingRect(coords)
                    # Extract the region of interest from the warped image
                    cropped_warped = warped_img[y:y+h, x:x+w]
                    # Save the cropped warped image
                    fliped= cv2.flip(cropped_warped, 0)  # Flip the image horizontally
                    # Save the cropped warped image
                    cv2.imwrite(os.path.join(ext_output_dir, f"{image_names[i]}.tiff"), fliped)
                    print(f"{image_names[i]}.tiff) at x:{x}, y:{y}, width:{w}, height:{h}")
                    rotation = np.eye(3).flatten()
                    rotation[8]=-1 
                    with open(cam_path, 'w') as f_out:
                        # First line: tx ty tz R00 R01 R02 R10 R11 R12 R20 R21 R22
                        line1 = f"{-1*(x+w/2)} {-(pano_height-y-h/2)} {w*fnormalize} " + \
                                " ".join(str(r) for r in rotation)
                        f_out.write(line1 + "\n")
                        # Second line: f d0 d1 paspect ppx ppy
                        line2 = f"{fnormalize} {d0} {d1} {paspect} {ppx} {ppy}"
                        f_out.write(line2 + "\n")


            #pano_width, pano_height = pano_size
            #print(f' ppy*height: {ppy*height} ppx*width: {ppx*width} pano_width: {pano_width} pano_height: {pano_height}')
            #print(f"dx: {H[0, 3]}, dy: {H[1, 3]}")
         
            # Invert the z-axis
            #H[2, 2] = -H[2, 2]  # Invert the z-axis

            #rot= H[:2, :2]  # Extract the rotation part
            #rot = np.linalg.inv(rot)  # Invert the rotation part
            #dx= H[0, 3] + (rot @ np.array([ppx * width, ppy * height]))[0]
            #dy= H[1, 3] + (rot @ np.array([ppx * width, ppy * height]))[1]
            #print(f"x: {( H[:2, :2] @ np.array([ppx * width, ppy * height]))[0]}")
            #print(f"y: {( H[:2, :2] @ np.array([ppx * width, ppy * height]))[1]}")
            #H[:2, :2] = np.linalg.inv(H[:2, :2])  # Invert the rotation part
            #H[0, 3] =  - dx 
            #H[1, 3] =  - dy
            #H[2, 3] = 0
            #flip_y = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])  # Flip Y-axis
            #cambio di base
            #H[:3, :3] =flip_y @ H[:3, :3]  # Apply the flip to the translation part
            # Get rotation and translation
            #translation = H[:3, 3]
            #rotation = H[:3, :3].flatten()
            #print(f"Translation: {translation}, Rotation: {rotation}")
            # invert z axis
          
            
def save_cam_images(
    image_names,
    image_grid,
    output_dir="texture",
    image_dir="",
    pano_width=11656, pano_height=8742, homography=None,
):

    for i in range(len(image_names)):
            img_name = image_names[i]
            base_name = os.path.splitext(os.path.basename(img_name))[0]
            ext_output_dir= os.path.join(image_dir,output_dir)
            if not os.path.exists(ext_output_dir):
                os.makedirs(ext_output_dir)
            H= np.array(homography[i]) 
            # revert the image
            img = image_grid[i]["image"]
            img = cv2.flip(img, 0)
            #img = cv2.flip(img, 1)
            cv2.imwrite(os.path.join(ext_output_dir, f"{base_name}.tiff"), img)
            #print(f"Saved: {base_name}.tiff")

def prepare_cam_folder(IMAGE_DIR, tm, indexes, folder_names,pano_size, type="albedo"):
    output_dir=f'texture_{type}'
    images_grid=loadImagesFromFolder(base_folder=IMAGE_DIR ,debug=False, loadSIFT=False,only_images=True,type=type)
    images_grid_flattened=[images_grid[r][c] for (r,c) in indexes]
    image_size = (images_grid[0][0]['image'].shape[1], images_grid[0][0]['image'].shape[0])
    homographie_flattened=[tm[r][c] for (r,c) in indexes]
    save_homographies_as_cam(homographie_flattened, folder_names, output_dir=output_dir,image_grid=images_grid_flattened,image_dir=IMAGE_DIR, width=image_size[0], height=image_size[1],pano_size=pano_size)
    #save_cam_images(folder_names,images_grid_flattened,output_dir=output_dir,image_dir=IMAGE_DIR,pano_width=pano_size[0],pano_height=pano_size[1],homography=tm)

def visualize_stitching_overlaps(global_homography, 
                               output_size=(1000, 1000), 
                               background_color=(240, 240, 240),
                               image_size=(11656, 8742)):
    """
    Enhanced visualization of stitched images with center labels and distinct colors.
    
    Args:
        global_homography: 2D grid of homography matrices
        output_size: Output image dimensions (width, height)
        background_color: Background color (B, G, R)
        image_size: Original image dimensions (width, height)
    
    Returns:
        Visualization image with colored regions and centered labels
    """
    # Create blank image
    vis_img = np.full((output_size[1], output_size[0], 3), background_color, dtype=np.uint8)
    
    # Get grid dimensions
    rows = len(global_homography)
    cols = len(global_homography[0]) if rows > 0 else 0
    
    # Define original image corners
    w, h = image_size
    original_corners = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ], dtype=np.float32)
    
    # Generate distinct colors for each image
    colors = []
    for i in range(rows):
        for j in range(cols):
            # Generate colors with good visual separation
            hue = int(180 * (i * cols + j) / (rows * cols))  # 0-179 range for HSV
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            colors.append((int(color[0]), int(color[1]), int(color[2])))
    
    # Draw each transformed image
    for i in range(rows):
        for j in range(cols):
            homography = global_homography[i][j]
            if homography is None:
                continue
            
            # Transform the corners
            transformed_corners = cv2.perspectiveTransform(
                original_corners.reshape(1, -1, 2), 
                np.array(homography))
            
            # Scale to output size (if needed)
            scale_x = output_size[0] / (w * 1)  # Adjust scaling factor as needed
            scale_y = output_size[1] / (h * 1)
            scaled_corners = transformed_corners * np.array([scale_x, scale_y])
            
            pts = scaled_corners.reshape(-1, 2).astype(np.int32)
            color_idx = i * cols + j
            
            # Fill the polygon with semi-transparent color
            overlay = vis_img.copy()
            cv2.fillPoly(overlay, [pts], colors[color_idx], lineType=cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.3, vis_img, 0.7, 0, vis_img)
            
            # Draw the border
            cv2.polylines(vis_img, [pts], True, colors[color_idx], 2, lineType=cv2.LINE_AA)
            
            # Calculate center point for label
            M = cv2.moments(pts)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = pts[0][0], pts[0][1]
            
            # Add label at center with contrasting color
            text_color = (0, 0, 0) if sum(colors[color_idx]) > 382 else (255, 255, 255)  # 382 = 255*1.5
            cv2.putText(vis_img, f"{i},{j}", (cX, cY), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    
    return vis_img