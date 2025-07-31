"""
Functions for computing and manipulating normal maps
"""
import re
import sys,os
from pathlib import Path
import numpy as np
import cv2
import trimesh
import json
import subprocess

import pymeshlab
#from PyQt5.QtWidgets import QMessageBox
#from PyQt5.QtCore import QCoreApplication
from scipy.spatial import Delaunay
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageCms,ImageOps

from core.photometric_stereo import PhotometricStereo
from core.photostereo import photometry
from utils.image_processing import lay_normals
from utils.depth_estimation import remove_bow_effect, depth_from_gradient_poisson, depth_from_gradient_torch
from utils.math_utils import check_bit_depth, check_decimation
from utils.image_processing import specularize_x
from utils.exif_tools import add_exif


from scipy.ndimage import binary_erosion, zoom
from skimage.morphology import disk
from skimage.color import rgb2lab, lab2rgb
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity

Image.MAX_IMAGE_PIXELS = None #Per evitare i warnings di pil


def load_processed_images(image_paths, options=None):
    """
    Load and process input images for photometric stereo.
    Python equivalent of PSLoadProcessedImagesTheMatlabWay.m
    
    Parameters
    ----------
    image_paths : list
        List of file paths to the input images
    options : dict
        Options for image loading:
            - image_channel: The image channel to use (0, 1, or 2)
            - resample: Whether to downsample images by factor of 10
            - normalize_percentile: If provided, images will be normalized to this percentile
        
    Returns
    -------
    ndarray
        Processed images as ndarray (height, width, num_images)
    """
    if options is None:
        options = {}
    
    # Set default options
    image_channel = options.get('image_channel', 0)  # Default to first channel (0 in Python)
    resample = options.get('resample', False)
    
    # Get image names and dimension
    num_images = len(image_paths)
    
    # Load first image to get dimensions
    #img = cv2.imread(image_paths[0], cv2.IMREAD_UNCHANGED)
    img_pil = Image.open(image_paths[0])
    img_pil.load()  # Load image data
    img = np.array(img_pil)
    if resample:
        # Calcola le nuove dimensioni
        new_width = int(img_pil.width * 0.1)
        new_height = int(img_pil.height * 0.1)
        # Ridimensiona l'immagine
        img_pil_resized = img_pil.resize((new_width, new_height), resample=Image.LANCZOS)
        img = np.array(img_pil_resized)
    
    # Get dimensions
    if len(img.shape) == 2:
        height, width = img.shape
        depth = 1
    else:
        height, width, depth = img.shape
    
    # Initialize output array
    I = np.zeros((height, width, num_images), dtype=np.float32)
    
    print(num_images)
    # Load images
    for i in range(num_images):
        print(f"Reading image: {i} of {num_images}", flush=True)

        # Read image and convert to double
        #img_tmp = cv2.imread(image_paths[i], cv2.IMREAD_UNCHANGED)
        with Image.open(image_paths[i]) as img_pil:
            img_pil.load()
            if resample:                    
                # Ridimensiona l'immagine
                img_pil_resized = img_pil.resize((width, height), resample=Image.LANCZOS)
                img_tmp = np.array(img_pil_resized)
            else:
                img_tmp = np.array(img_pil)

        #print(img_tmp.shape)
        #print(img_tmp2.shape)
        img_tmp = img_tmp.astype(np.float32)
        
        # Determine divider based on bit depth
        if np.max(img_tmp) > 1000:
            divider = 65536.0
        else:
            divider = 255.0
        
        # Normalize to [0, 1]
        img_tmp = img_tmp / divider
        
        # Check if we need to select a specific channel
        if depth > 1 and image_channel < depth:
            # The MATLAB code does img_tmp(end:-1:1, :, options.ImageChannel)
            # This flips the image vertically and selects a specific channel
            img_tmp = np.flip(img_tmp, axis=0)
            img_tmp = img_tmp[:, :, image_channel]


        # Apply percentile normalization if requested
        if 'normalize_percentile' in options:
            pct = np.percentile(img_tmp, options['normalize_percentile'])
            if pct > 0:  # Avoid division by zero
                img_tmp = img_tmp / pct
        
        # Store the processed image
        I[:, :, i] = img_tmp
    
    return I

def reorder_lights_matrix(config, matrice_luci):
    if config['DDLightsOrder'] == '45W...15S':
        matrice_temp = matrice_luci
        matrice_luci[:,0]=matrice_temp[:,3]
        matrice_luci[:,1]=matrice_temp[:,0]
        matrice_luci[:,2]=matrice_temp[:,1]
        matrice_luci[:,3]=matrice_temp[:,2]
        matrice_luci[:,4]=matrice_temp[:,7]
        matrice_luci[:,5]=matrice_temp[:,4]
        matrice_luci[:,6]=matrice_temp[:,5]
        matrice_luci[:,7]=matrice_temp[:,6]
    return matrice_luci

# def set_nans_to_zero(array_in):
#     return np.where(np.isnan(array_in), 0, array_in)
#     # mask = array_in[np.isnan(array_in)]
#     # return cv2.inpaint(array_in, mask, 3, cv2.INPAINT_TELEA)


def compute_normal_maps_new(config :dict, progress_callback=None):
    """
    Compute normal map from images using photometric stereo
    
    Args:
        images: List of input images
        light_directions: List of light direction vectors
        method: Method to use for normal map computation
        
    Returns:
        tuple: (normal_map, albedo)
    """
    if progress_callback:
        progress_callback.emit(0)
        #QCoreApplication.processEvents()
    
    #Image.MAX_IMAGE_PIXELS = 933120000

    lights_file_path = config['lights_file_path']
    matrice_luci = np.loadtxt(lights_file_path)
    matrice_luci = reorder_lights_matrix(config, matrice_luci)

    # Placeholder implementation
    if not config['light_direction_images'] or not config['all_lights_on_image']:
        raise ValueError("Images and light directions must not be empty")
    
    # Load images
    #print(config['light_direction_images'])
    images = load_processed_images(config['light_direction_images'], config['loadOptions'])
    n_imgs = images.shape[2]

    if progress_callback:
        progress_callback.emit(10)
        #QCoreApplication.processEvents()
        print(f"Reading images complete", flush=True)

    print("Creating a shadow mask", flush=True)
    # Create a shadow mask
    shadow_mask = images > 0.05  # Equivalent to MATLAB's threshold
    shadow_mask = shadow_mask.astype(int)  # Convert to binary mask
    se = disk(config['StrelSize'])  # Structuring element (disk)

    if progress_callback:
        progress_callback.emit(30)
        #QCoreApplication.processEvents()


    # Estimate normal vectors and albedo with light strength estimation
    print("Estimating normal vectors and albedo (with light strength estimation)...", flush=True)
    lambda_values = np.loadtxt(config['light_strenght_file_path'])  # Load light strength
    matrice_luci = (lambda_values.reshape(1, -1) * matrice_luci)  # Scale light directions by strength

    ps = PhotometricStereo(images, matrice_luci, shadow_mask)
    albedo, normals = ps.process_matlab()
    normals[:, :, 0] = np.nan_to_num(normals[:, :, 0], 0.0)
    normals[:, :, 1] = np.nan_to_num(normals[:, :, 1], 0.0)
    normals[:, :, 2] = np.nan_to_num(normals[:, :, 2], 0.5)

    print("Normals and albedo computed", flush=True)

    # Straighten normal positions
    # Ha senso (???)
    normals = lay_normals(normals)
    print("Evaluating normals orientation", flush=True)
    if progress_callback:
        progress_callback.emit(50)
        #QCoreApplication.processEvents()

    #MAP2
    results = compute_maps_step(config, normals, albedo, 1, progress_callback)
    normals_step1 = results['normals']
    
    if progress_callback:
        progress_callback.emit(75)
        #QCoreApplication.processEvents()
    
    # Step 2
    results = compute_maps_step(config, normals_step1, albedo, 2, progress_callback)
    
    if progress_callback:
        progress_callback.emit(100)
        #QCoreApplication.processEvents()
    
    print('*** Done. ***')
    return results


    
    #return normal_map, albedo




def eval_n_estimate_by_i_error(rho, n, I, mask, light_directions, options=None):
    """
    Evaluate scaled normal estimation by intensity error.
    Python equivalent of EvalNEstimateByIError.m
    
    Parameters
    ----------
    rho : ndarray
        Albedo map, shape (height, width)
    n : ndarray
        Normal map, shape (height, width, 3) or (3, height, width)
    I : ndarray
        Input images, shape (height, width, num_images)
    mask : ndarray
        Shadow mask, shape (height, width, num_images)
    light_directions : ndarray
        Light directions, shape (3, num_images)
    options : dict, optional
        Options:
        - display: whether to print error statistics
        
    Returns
    -------
    ndarray
        Error map, shape (height, width)
    """
    if options is None:
        options = {'display': False}
    
    # Handle different dimension layouts for n
    if n.shape[0] == 3:
        n = np.transpose(n, (1, 2, 0))
    
    # Get dimensions
    height, width, num_images = I.shape
    N = height * width
    
    # Resize (vectorize) the input
    I_reshaped = I.reshape(N, num_images)
    mask_reshaped = mask.reshape(N, num_images)
    n_reshaped = n.reshape(N, 3)
    
    # Calculate b = rho * n
    rho_expanded = np.repeat(rho.reshape(N, 1), 3, axis=1)
    b = rho_expanded * n_reshaped
    
    # Compute error map
    Ierr = np.zeros(N)
    for i in range(num_images):
        Ierr_i = I_reshaped[:, i] - np.dot(b, light_directions[:, i])
        Ierr_i[~mask_reshaped[:, i]] = 0
        Ierr = Ierr + Ierr_i**2
    
    # Compute RMS and reshape
    mask_sum = np.sum(mask_reshaped, axis=1)
    
    # Avoid division by zero
    mask_sum[mask_sum == 0] = np.nan
    Ierr = np.sqrt(Ierr / mask_sum)
    Ierr = Ierr.reshape(height, width)
    
    # Print error statistics
    if options.get('display', False):
        Ierr_valid = Ierr[np.isfinite(Ierr)]
        print('Evaluate scaled normal estimation by intensity error:')
        print(f'  RMS = {np.sqrt(np.mean(Ierr_valid**2)):.4f}')
        print(f'  Mean = {np.mean(Ierr_valid):.4f}')
        print(f'  Median = {np.median(Ierr_valid):.4f}')
        print(f'  90 percentile = {np.percentile(Ierr_valid, 90):.4f}')
        print(f'  Max = {np.max(Ierr_valid):.4f}')
    
    return Ierr


# def downscale_depth_map(Z, scale=0.10):
#     Z = cv2.resize(Z, None, fx = scale, fy = scale)
#     return Z


def compute_maps_step(config, normals, albedo, step, progress_callback=None):
    """
    Equivalent to PSBoxComputeMaps2 function.
    Process normals and albedo in two steps.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    normals : ndarray
        Normal map, shape (height, width, 3)
    albedo : ndarray
        Albedo map, shape (height, width)
    step : int
        Step number (1 or 2)
    progress_callback : function, optional
        Progress callback function
        
    Returns
    -------
    dict
        Dictionary with processed maps
    """
    downsample = config['image_downsample']

    # Subtract black error from normals
    black_error_path = config.get('black_error') if step == 1 else config.get('black_error_2')
    if black_error_path and os.path.exists(black_error_path):
        #err_image = cv2.imread(black_error_path, cv2.IMREAD_UNCHANGED)
        print(f"Loading residual image {black_error_path}", flush=True)
        err_image = Image.open(black_error_path)
        err_image.load()  # Load image data

        # if downsample:
        #     #err_image = cv2.resize(err_image, (normals.shape[1], normals.shape[0]))
        #     err_image = err_image.resize((normals.shape[1], normals.shape[0]), resample=Image.LANCZOS)

        # for debugging purposes only
        if normals.shape[0] != err_image.height or normals.shape[1] != err_image.width:
            # err_image = cv2.resize(err_image, normals.shape[1], normals.shape[0], resample=Image.LANCZOS)
            err_image = err_image.resize((normals.shape[1], normals.shape[0]), Image.Resampling.LANCZOS )

        err_image = np.array(err_image)

        # Determine bit depth for normalization
        divider = check_bit_depth(err_image)

        # Apply error correction
        err_image_float = err_image.astype(np.float32) / divider

        if len(err_image_float.shape) == 2:
            # If grayscale, apply to all channels
            #print("Neri disabilitati")
            #Disabilito temporaneamente i NERI
            normals = normals - err_image_float[:, :, np.newaxis]
        else:

            normals = normals - err_image_float
                                                                                                                                                                                                                                                                                                                                                                   
    # Estimate depth map from normal vectors
    print('Estimating normal vectors...', flush=True)

    # Results dictionary
    results = {
        'normals': normals,
        'albedo': albedo
    }

    try:
        if step == 1:
            print('Processing step 1...', flush=True)
            print("Analyzing and correcting normals bow effect", flush=True)
            normals_corrected = remove_bow_effect(normals, sigma=60)
   
            # Convert normals to RGB for visualization
            print("Rendering normals in RGB", flush=True)
            normals_rgb = render_normals_in_rgb(normals_corrected)
            results['normals'] = normals_corrected # Per rimetterlo in linea con matlab, rimettere results['normals'] = normals
            results['normals_rgb'] = normals_rgb
        
            # Save normals if configured
            if 'output_directory' in config:
                print("Writing normals in RGB", flush=True)
                write_normals(normals_rgb, config['output_directory'], config)
            return results  # Return normals for step 2
    
        elif step == 2:
            print('Processing step 2...', flush=True)
            # Process all lights on image if available
            if 'all_lights_on_image' in config and os.path.exists(config['all_lights_on_image']):
                #all_lights_on_image = cv2.imread(config['all_lights_on_image'], cv2.IMREAD_UNCHANGED)
                full_image_path = config['all_lights_on_image']
                print(f"Reading fully lit image: {full_image_path}", flush=True)
                all_lights_on_image = Image.open(full_image_path)
                all_lights_on_image.load()  # Load image data
                all_lights_on_image = np.array(all_lights_on_image)
                divider = check_bit_depth(all_lights_on_image)
                all_lights_on_image = (all_lights_on_image.astype(float) / divider)
                
                if downsample:
                    #all_lights_on_image = cv2.resize(all_lights_on_image, (albedo.shape[1], albedo.shape[0]))
                    print(f"Resampling fully lit image {full_image_path} (--image-downsample -ig is True)", flush=True)
                    all_lights_on_image = all_lights_on_image.resize((albedo.shape[1], albedo.shape[0]), resample=Image.Resampling.LANCZOS)

                all_lights_on_image = np.array(all_lights_on_image)
            
                # Save albedo if configured
                if 'output_directory' in config:
                    write_albedo_tiff(config,all_lights_on_image, albedo)
                    # Save reflection map
                    write_reflection_map(config,all_lights_on_image, albedo)
        
            # Compute mesh and save depth map
            if 'output_directory' in config:

                print("Computing depth map with Poisson solver", flush=True)
                Z = depth_from_gradient_poisson(normals)
                #Z = depth_from_gradient_torch(normals, device='cuda') # out of memory
                Z -= np.min(Z)

                print("Computing vertices and faces from the depth map", flush=True)
                vertices, faces = compute_nodes_and_faces(Z)
                results['depth '] = Z

                print("Generating the mesh", flush=True)
                mesh_tr = trimesh.Trimesh(vertices=vertices, faces=faces)

                print("Winding is consistent:", mesh_tr.is_winding_consistent, flush=True)
                if not mesh_tr.is_winding_consistent:
                    print("Winding is not consistent. Trying to fix normals", flush=True)
                    mesh_tr.fix_normals()

                #Richiamo meshlab solo se sto lavorando sulle immagini full res
                if not downsample:
                    print("Performing anisotropic remeshing", flush=True)
                    mesh_tr = remesh_trimesh_with_pymeshlab(mesh_tr,target_percentage=0.5,iterations=10)

                mesh_filename = get_depth_map_output_path(config)
                print(f"Saving mesh to .ply file: {mesh_filename}", flush=True)
                mesh_tr.export(mesh_filename)
                mesh = {'vertices': vertices, 'faces': faces}

    except BaseException as e:
        print(f"Error : {e}")
        return None
    
    return results

def extract_json_from_stdout(stdout):

    match = re.search(r'(\{.*\})', stdout, re.DOTALL)
    if match:
        json_str = match.group(1)
        result = json.loads(json_str)
        return result
    else:
        raise RuntimeError(f"Output from pymeshlab does not contain JSON. STDOUT: {stdout}")

def remesh_trimesh_with_pymeshlab(mesh_tm, target_percentage=0.25, iterations=10):
    ms = pymeshlab.MeshSet()
    mesh_pl = pymeshlab.Mesh(
        vertex_matrix=np.asarray(mesh_tm.vertices, dtype=np.float32),
        face_matrix=np.asarray(mesh_tm.faces, dtype=np.int32)
    )

    ms.add_mesh(mesh_pl, "input_from_trimesh")

    ms.meshing_isotropic_explicit_remeshing(
        iterations=iterations,
        adaptive=True,
        targetlen=pymeshlab.PercentageValue(target_percentage),
        collapseflag=True,
        swapflag=True,
        smoothflag=True,
        reprojectflag=True
    )


    remeshed = ms.current_mesh()
    vertices_remeshed = remeshed.vertex_matrix()
    faces_remeshed = remeshed.face_matrix()

    return trimesh.Trimesh(vertices=vertices_remeshed, faces=faces_remeshed)


def call_pymeshlab_backend(config, vertices, faces):

    current_path = Path.cwd().resolve()
    python_exe = Path(config.get("pymeshlab_virtual_environment_absolut_path")) #python_exe = pyMeshLabPath / ".venv" / "Scripts" / "python.exe"
    script_path = Path("")

    if not os.path.exists(python_exe):
        raise FileNotFoundError("Pymeshlab Exception: " + str(python_exe))

    proc = subprocess.Popen(
        [str(python_exe), str(script_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    print("Starting pymeshlab process...")
    print(f'Python path: {python_exe}')
    print(f'Script path: {script_path}')
    path_out = get_depth_map_output_path(config)
    downsample = config['image_downsample']
    try:
        request = json.dumps({"vertices": vertices, "faces": faces, "path_out": path_out, "downsample": False})
        stdout, stderr = proc.communicate(input=request, timeout=2000)

        # print(f"STDOUT: {stdout}")
        # print(f"STDERR: {stderr}")

        # result = extract_json_from_stdout(stdout)
        # print(f"Result: {result}")
        if stderr == "":
            print("ok")
            # v_out = result["mesh"]["vertices"]
            # f_out = result["mesh"]["faces"]
        else:
            raise ValueError("Pymeshlab Error: " + stdout["message"])
    except BaseException as e:
        print("ko " + str(e))
        raise RuntimeError("Pymeshlab Exception: " + str(e))






def write_normals(normals_rgb, output_path, config):
    """
    Write normals RGB image to file
    Equivalent to writeNormals function in MATLAB
    
    Parameters
    ----------
    normals_rgb : ndarray
        RGB representation of normal map
    output_path : str
        Path to save the normal map image
    """

    image_name = get_image_name_only(config)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save image
    #rgb_path_out = os.path.join(output_path, f"{image_name}-PolynomialCorrectedNormals-RGB.png")
    #cv2.imwrite(rgb_path_out, cv2.cvtColor(normals_rgb, cv2.COLOR_RGB2BGR))
    #☺print(f"Saved normal map RGB to {rgb_path_out}")

    print("*** Writing polynomial corrected normals ***")
    # Costruisci il percorso completo per il file di output
    full_path_out = os.path.join(output_path, f"{image_name}-PolynomialCorrectedNormals.tif")

    # Applica specularize_x e salva l'immagine
    normals_rgb_specularized = specularize_x(normals_rgb)
    #print(normals_rgb_specularized.shape)
    #cv2.imwrite(full_path_out, cv2.cvtColor(normals_rgb_specularized, cv2.COLOR_RGB2BGR))
    image_out = Image.fromarray(normals_rgb_specularized, mode='RGB')
    image_out.save(full_path_out, compression='tiff_lzw')

    #image_out = Image.fromarray(normals_rgb_specularized, mode='RGB')
    #image_out.save(full_path_out, compression='tiff_lzw')

def write_reflection_map(config, all_lights_on_image, rho):
    """
    Write the reflection map to a file.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing:
            - 'OutputFolder': Path to the output folder.
    all_lights_on_image : ndarray
        Fully lit image as an RGB array.
    rho : ndarray
        Albedo map as a 2D array.

    Returns
    -------
    None
    """
    # Get the image name
    image_name = get_image_name_only(config)
    print('*** Reading fully lit image ***')

    full_path_out = os.path.join(config['output_directory'], f"{image_name}-ReflectionMap.tif")

    # Compute the reflection map
    #all_lights_gray = rgb2gray(all_lights_on_image)
    '''
    max_rho = np.max(rho)
    if max_rho > 0:
        rho_norm = rho / max_rho
    else:
        rho_norm = np.zeros_like(rho)
    '''
    all_lights_gray = rgb2gray(all_lights_on_image)
    #reflection_map = im2double(all_lights_gray - specularize_x(rho / (np.mean(rho) * 2.0)))
    reflection_map = all_lights_gray - specularize_x(rho / np.max(rho))
    reflection_map = rescale_intensity(reflection_map, in_range='image', out_range=(0, 1))
    '''
    if np.max(reflection_map) > 0:
        reflection_map = reflection_map / np.max(reflection_map)
    else:
        reflection_map = np.zeros_like(reflection_map)
    '''
    print('*** Writing reflection map ***')
    #image_out = Image.fromarray(normals_rgb_specularized, mode='RGB')
    #image_out.save(full_path_out, compression='tiff_lzw')
    reflection_map_uint8 = (reflection_map * 255).astype(np.uint8)
    cv2.imwrite(full_path_out, reflection_map_uint8)


def compute_nodes_and_faces(depth_map, smoothing=False, sigma=1.5, vertices_scale=1):
    h, w = depth_map.shape

    # Eventuale Filtro smoothing per ridurre rumore mantenendo dettagli

    if smoothing:
        alpha = 0.7
        depth_map_smooth = gaussian_filter(depth_map, sigma=1.5)
        depth_map = alpha * depth_map + (1 - alpha) * depth_map_smooth

    # Normalizzazione robusta e scala
    depth_map = depth_map - depth_map.min()
    depth_map = depth_map / (np.percentile(depth_map, 99) + 1e-8)
    depth_map = depth_map * 10

    y, x = np.mgrid[0:h, 0:w]
    vertices = np.column_stack((x.flatten(), y.flatten(), depth_map.flatten())).astype(np.float32)
    if vertices_scale != 1:
        vertices *= vertices_scale

    faces = []
    for i in range(h - 1):
        for j in range(w - 1):
            v0 = i * w + j
            v1 = v0 + 1
            v2 = v0 + w
            v3 = v2 + 1
            # due triangoli per quadratp
            # faces.append([v0, v2, v1])
            # faces.append([v1, v2, v3])
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])

    faces = np.array(faces, dtype=np.int32)

    return vertices, faces

# def compute_nodes_and_faces(depth_map):
#     """
#     Compute mesh nodes and faces from depth map
#     Equivalent to computeNodesAndFaces function in MATLAB
#
#     Parameters
#     ----------
#     depth_map : ndarray
#         Depth map
#
#     Returns
#     -------
#     tuple
#         (nodes, faces) - nodes are 3D points, faces are triangle indices
#     """
#     depth_map = (depth_map - depth_map.min()) * 10
#
#
#     h, w = depth_map.shape
#     y, x = np.mgrid[0:h, 0:w]
#     vertices = np.column_stack((x.flatten(), y.flatten(), depth_map.flatten())).astype(np.float32)
#
#     faces = []
#     for i in range(h - 1):
#         for j in range(w - 1):
#             v0 = i * w + j
#             v1 = v0 + 1
#             v2 = v0 + w
#             v3 = v2 + 1
#             faces.append([v0, v2, v1])
#             faces.append([v1, v2, v3])
#
#     faces = np.array(faces, dtype=np.int32)
#
#     return vertices, faces

# def compute_nodes_and_faces(depth_map):
#
#     h, w = depth_map.shape
#
#     # Filtra per togliere rumore ma conserva dettagli
#     depth_map_smooth = gaussian_filter(depth_map, sigma=1)
#     alpha = 0.7
#     depth_map = alpha * depth_map + (1 - alpha) * depth_map_smooth
#
#     # Scala in modo robusto
#     depth_map = depth_map - depth_map.min()
#     depth_map = depth_map / (np.percentile(depth_map, 99) + 1e-8)
#     depth_map = depth_map * 10
#
#     y, x = np.mgrid[0:h, 0:w]
#     points2D = np.column_stack((x.flatten(), y.flatten()))
#     vertices = np.column_stack((x.flatten(), y.flatten(), depth_map.flatten())).astype(np.float32)
#
#     tri = Delaunay(points2D)
#     faces = tri.simplices.astype(np.int32)
#
#     return vertices, faces



def get_depth_map_output_path(config):
    image_name = get_image_name_only(config)
    if config['height_map_format'].lower() == 'ply':
        fp = os.path.join(config['output_directory'], f"{image_name}-DepthMap.ply")
        #write_surface_mesh(mesh_out, fp, encoding='binary')
        #write_surface_mesh_in_meshlab(mesh_out, fp, encoding='binary')
    else:
        fp = os.path.join(config['output_directory'], f"{image_name}-DepthMap.stl")
    return fp




# spostata nel progetto meshalb
# def write_depth_map(config, mesh_out, vertices):
#     """
#     Write the depth map to a file in PLY or STL format.
#
#     Parameters
#     ----------
#     config : dict
#         Configuration dictionary containing:
#             - 'OutputFolder': Path to the output folder.
#             - 'DDHeightMapExport': Export format ('ply' or other).
#     mesh_out : dict
#         Mesh data containing at least 'faces' and 'vertices'.
#
#     Returns
#     -------
#     None
#     """
#     # Get the image name
#     image_name = get_image_name_only(config)
#
#     if config['height_map_format'].lower() == 'ply':
#         print('*** Writing PLY ***')
#         fp = os.path.join(config['output_directory'], f"{image_name}-DepthMap.ply")
#         #write_surface_mesh(mesh_out, fp, encoding='binary')
#         #write_surface_mesh_in_meshlab(mesh_out, fp, encoding='binary')
#     else:
#         print('*** Writing STL ***')
#         fp = os.path.join(config['output_directory'], f"{image_name}-DepthMap.stl")
#         write_stl(mesh_out['faces'],vertices, fp)


def get_image_name_only(config):
    """
    Extract the image name from the configuration.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing:
            - 'InputImagePath': Path to the input image.

    Returns
    -------
    str
        The base name of the input image without extension.
    """
    input_image_path = config.get('all_lights_on_image', '')
    return os.path.splitext(os.path.basename(input_image_path))[0]


def render_normals_in_rgb(nn):
    """
    Convert normal vectors to an RGB representation.

    Parameters
    ----------
    nn : ndarray
        Input normal vectors. Can be a 1D array of size 3 or a 3D array (height, width, 3).

    Returns
    -------
    rgb : ndarray
        RGB representation of the normal vectors as an 8-bit unsigned integer array.
    """
    rgb = np.zeros_like(nn, dtype=np.float32)
    
    # Normal map (3D array)
    rgb[:, :, 0] = np.round((nn[:, :, 0] + 1) / 2 * 255)
    rgb[:, :, 1] = np.round((nn[:, :, 1] + 1) / 2 * 255)
    rgb[:, :, 2] = np.round(nn[:, :, 2] * 255)
    
    return rgb.astype(np.uint8)


def write_albedo_tiff(config, all_lights_on_image, rho):
    """
    Write the normalized albedo to a file.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing:
            - 'OutputFolder': Path to the output folder.
            - 'allLightsOnImagePath': Path to the "all lights on" image.
    all_lights_on_image : ndarray
        The "all lights on" image as an RGB array.
    rho : ndarray
        The albedo map as a 2D array.

    Returns
    -------
    None
    """
    # Get the image name
    image_name = get_image_name_only(config)
    print("*** Writing normalized Albedo ***")

    # Build the output file path
    full_path_out = os.path.join(config['output_directory'], f"{image_name}-L-albedo.tif")
    #full_path_out_png = os.path.join(config['output_directory'], f"{image_name}-L-albedo.png")

    # Convert the "all lights on" image to LAB and normalize
    #all_lights_lab = specularize_x(rgb2lab(all_lights_on_image))
    all_lights_lab = specularize_x(cv2.cvtColor(np.float32(all_lights_on_image), cv2.COLOR_RGB2Lab))

    # Clip and normalize the albedo
    # and get it as the L channel
    #L_rho = np.array(np.clip((rho / (np.mean(rho) * 2.0)), 0, 1)) * 100
    L_rho = np.array(np.clip((rho / np.max(rho)), 0, 1)) * 100
    # a and b channels are from the fully lit image
    a_all_lights = np.array(all_lights_lab[:, :, 1])
    b_all_lights = np.array(all_lights_lab[:, :, 2])
    reconstructed_albedo_lab = np.float32(np.dstack((L_rho, a_all_lights, b_all_lights)))

    #rho_n = lab2rgb(np.stack((L_rho * 100, a_all_lights, b_all_lights), axis=2))
    rho_n = cv2.cvtColor(reconstructed_albedo_lab, cv2.COLOR_Lab2RGB)

    # Save the normalized albedo as a TIFF file 
    #rho_n[np.isnan(rho_n)] = 0
    rho_n_uint8 = specularize_x((rho_n * 255).astype(np.uint8))


    #Uso pil anziché cv2, per comprimere il tif in lzw
    image_out = Image.fromarray(rho_n_uint8, mode='RGB')
    image_out.save(full_path_out, compression='tiff_lzw')
    #cv2.imwrite(full_path_out_png, cv2.cvtColor(rho_n_uint8, cv2.COLOR_RGB2BGR))

    # Add the Display P3 color profile
    add_display_p3_profile(config['all_lights_on_image'], full_path_out)


def add_display_p3_profile(image_containing_the_profile_to_copy, full_path_out):
    """
    Add a Display-P3 or sRGB profile to the output image based on the profile of the input image.

    Parameters
    ----------
    image_containing_the_profile_to_copy : str
        Path to the image containing the ICC profile to copy.
    full_path_out : str
        Path to the output image where the profile will be added.

    Returns
    -------
    None
    """
    is_profiled = False
    img_profile = None

    try:
        # Attempt to read the ICC profile from the input image
        img_profile = ImageCms.getOpenProfile(image_containing_the_profile_to_copy)
        is_profiled = True
    except Exception as e:
        print(f"Warning: no ICC profile found in: {image_containing_the_profile_to_copy}. Error: {e}")

    if is_profiled:
        if hasattr(img_profile, 'product_desc') and 'appl' in img_profile.product_desc.lower():
            # Add Display-P3 profile
            add_exif(image_containing_the_profile_to_copy, full_path_out, 'DISPLAY-P3')
        elif hasattr(img_profile, 'product_desc') and 'srgb' in img_profile.product_desc.lower():
            # Add sRGB profile
            add_exif(image_containing_the_profile_to_copy, full_path_out, 'sRGB')
        else:
            # Add no profile
            add_exif(image_containing_the_profile_to_copy, full_path_out, 'none')
    else:
        # Add no profile if no ICC profile is found
        add_exif(image_containing_the_profile_to_copy, full_path_out, 'none')

def im2double(image):
    """
    Convert an image to double precision and normalize to the range [0, 1].

    Parameters
    ----------
    image : ndarray
        Input image (e.g., uint8, uint16, or float).

    Returns
    -------
    ndarray
        Normalized image in double precision.
    """
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    elif image.dtype == np.uint16:
        return image.astype(np.float32) / 65535.0
    elif image.dtype == np.float32:
        return image  # Already in double precision
    elif image.dtype == np.float64:
        return image.astype(np.float32)

    else:
        raise ValueError("Unsupported image data type.")


