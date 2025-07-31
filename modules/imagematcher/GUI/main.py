import os
# from PIL import Image
# import numpy as np
import cv2
# import sys
# import os
import argparse
import copy

from numpy import log
# import decimal
from components.StitchGUI import SIFTMatcherApp
from core.GridMatcher import match_adjacent_images_from_center_extended
from core.ImageGridStitcher import ImageGridStitcher
from core.IO import *
import logging
import datetime
from components.PreviewImage import PreviewImage
from PIL import Image, ImageTk
from core.MeshFusion import MeshFusion
now = datetime.datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
logger = logging.getLogger("Test Stiching")
#logging.basicConfig(filename=f'log/Test-execution-{timestamp}.log', level=logging.INFO)

cv2.ocl.setUseOpenCL(False)
# Read from arguments
parser = argparse.ArgumentParser()
parser.add_argument("input", nargs=1, help='Input file')
parser.add_argument("--m", choices=['au', 'ma', 'fi','all'], help='Mode of operation: au (automatic), ma (manual), fi (final)')
parser.add_argument("--prv", action='store_true', help='Stitch preview')
# Bundle adjustment
parser.add_argument("--ba", action='store_true', default=True, help='Use bundle adjustment')
parser.add_argument("--not_ba", action='store_false', default=False ,help='Do not use bundle adjustment')

# SIFT parameters
parser.add_argument("--snf", type=int, default=0, help='Number of features to use in SIFT')
parser.add_argument("--sol", type=int, default=3, help='Number of octave layers to use in SIFT')
parser.add_argument("--sct", type=float, default=0.04, help='Contrast threshold to use in SIFT')
parser.add_argument("--set", type=float, default=10, help='Edge threshold to use in SIFT')
parser.add_argument("--ss", type=float, default=1.6, help='Sigma to use in SIFT')
# Match parameters
parser.add_argument("--md", type=float, default=0.8, help='Match distance to use in matching')
parser.add_argument("--mi", type=int, default=15, help='Minimum inliers to use in matching')
parser.add_argument("--rt", type=float, default=1.0, help='RANSAC threshold to use in matching')
# Load SIFT from file
parser.add_argument("--lsf", type=str, default='', help='Path to load SIFT from file')
parser.add_argument("--lmf", type=str, default='', help='Path to load matches from file')
# Stitch options
parser.add_argument("--sr", action='store_true', help='Stitch reflections')
parser.add_argument("--sn", action='store_true', help='Stitch normals')
parser.add_argument("--sm", action='store_true', help='Stitch mesh')
# bundle adjustment options
parser.add_argument("--bnp", type=int, default=150, help='Numbers of bundle parameters to optimize for each pair of images')

args = parser.parse_args()
IMAGE_DIR= args.input[0]
mode = args.m
is_load_sift_from_file = args.lsf != ''
is_load_matches_from_file = args.lmf != ''
is_stitch_reflections = args.sr
is_stitch_normals = args.sn
is_stitch_mesh = args.sm
sift_file_path = args.lsf if args.lsf != '' else os.path.join(IMAGE_DIR, "sift_features.hdf5")
match_file_path = args.lmf if args.lmf != '' else os.path.join(IMAGE_DIR, "match_list.hdf5")
is_stitch_preview = args.prv

def get_unique_output_name(base_name, suffix, extension):
    i = 1
    output_name = f"{base_name}_{suffix}_{i}.{extension}"
    while os.path.exists(output_name):
        i += 1
        output_name = f"{base_name}_{suffix}_{i}.{extension}"
    return output_name

siftoptions = {
        "nfeatures": args.snf,
        "nOctaveLayers": args.sol,
        "contrastThreshold": args.sct,
        "edgeThreshold": args.set,
        "sigma": args.ss
    }
if (mode == 'au' or mode == 'all'):
    print(f"---[LOADING IMAGES FROM FOLDER]---")
    if not is_load_sift_from_file:
        images_grid=loadImagesFromFolder(base_folder=IMAGE_DIR ,debug=True, loadSIFT=True, siftoptions=siftoptions)
        save_images_grid_to_file(sift_file_path, images_grid)
    else:
    # print ("loading sift from file")
        logging.info("loading sift from file")
        images_grid = restore_images_grid(sift_file_path)
    rows = len(images_grid)
    # the minimun lenght of image grid element
    cols = min([len(r)for r in images_grid])
    if not is_load_matches_from_file:
        print(f"---[MATCHING IMAGES]---")
        matches_list = match_adjacent_images_from_center_extended(images_grid=images_grid, rows=rows, cols=cols, match_distance=args.md, min_inliers=args.mi, ransac_threshold=args.rt)
        # salvare i match info per immagine
        save_matches_to_h5(match_file_path, matches_list)
    else:
        # print ("loading matches from file")
        logging.info("loading matches from file")
        matches_list = load_matches_from_h5(match_file_path)
        # append also all flags on the name
    flags = ""
    # if args.ba:
    #     flags += "_ba"
    output_name = get_unique_output_name(os.path.join(IMAGE_DIR, os.path.basename(IMAGE_DIR)), flags, "tiff")
    is_ba = True
    if args.not_ba:
        is_ba = False
    print(f"---[MERGING IMAGE]---")
    stitcher = ImageGridStitcher(images_grid, matches_list)
    merged_image = stitcher.get_stitched_image_from_center(save=False, image_name=output_name, rows=rows, cols=cols, ba=is_ba, ba_num_params=args.bnp)
    output_name = get_unique_output_name(os.path.join(IMAGE_DIR, os.path.basename(IMAGE_DIR)), "merged", "tiff")
    cv2.imwrite(output_name, merged_image)
    print(f"---[SAVING IMAGE {output_name}]---")
    gh=stitcher.getGlobalHomographies()
    pano_y, pano_x= stitcher.getPanoramaSize()
    output_name = get_unique_output_name(os.path.join(IMAGE_DIR, os.path.basename(IMAGE_DIR)), flags, "tiff")
    save_global_homography_to_h5(os.path.join(IMAGE_DIR, "global_homography.hdf5"), gh,(pano_x, pano_y))
    save_global_homography_to_txt(os.path.join(IMAGE_DIR,"gh.txt"),gh,(pano_x, pano_y))
    if is_stitch_preview:
        # Convert merged_image to PIL Image
        merged_image_pil = Image.fromarray(cv2.cvtColor(merged_image, cv2.COLOR_BGR2RGB))
        # Preview the image
        preview= PreviewImage(merged_image_pil, output_name)
        preview.run()
    visualization= visualize_stitching_overlaps(gh,(pano_x,pano_y))
    visualization_name="stitching_overlaps_visualization.png"
    cv2.imwrite(os.path.join(IMAGE_DIR,visualization_name), visualization)

    #### Apri preview
if mode == 'ma':
    # Create and run the application
    IMAGE_DIR=args.input[0]
    images_grid = restore_images_grid(sift_file_path)
    matches_list = load_matches_from_h5(match_file_path)
    app = SIFTMatcherApp(matches_list,images_grid,IMAGE_DIR, ransac_threshold=5.0, bundle_adjustment=args.ba, siftoptions=siftoptions)
    app.run()

if (mode == 'fi' or mode == 'all'):
    # restore image grid with decttion (sift points) info 
    images_grid = restore_images_grid(sift_file_path)
    # restore grid information aboout matches 
    matches_list = load_matches_from_h5(match_file_path)
    gh, final_pano_size = load_global_homo_from_h5(os.path.join(IMAGE_DIR, "global_homography.hdf5"))
    print(f"final_pano_size {final_pano_size}")
    rows = len(images_grid)
    cols = min([len(r)for r in images_grid])
    # initiate the stitcher with gray images 
    stitcher = ImageGridStitcher(images_grid, matches_list)
    stitcher.setGlobalHomography(gh)
    print(f"---[MERGING ALBEDO IMAGE]---")
    images_grid=loadImagesFromFolder(base_folder=IMAGE_DIR ,debug=False, loadSIFT=False,only_images=True,type="albedo")
    albedo = stitcher.stitch_images_rgb(images_grid)
    output_name = get_unique_output_name(os.path.join(IMAGE_DIR, os.path.basename(IMAGE_DIR)), "albedo", "tiff")
    cv2.imwrite(output_name, albedo)
    print(f"---[SAVING IMAGE {output_name}]---")
    ### Aggiungere Leggere da file
    IMAGE_DIR=args.input[0]
    image_size = (images_grid[0][0]['image'].shape[1], images_grid[0][0]['image'].shape[0])
    #print(f"image_size {image_size}")
    if is_stitch_normals:
        print(f"---[MERGING NORMALS IMAGE]---")
        images_grid=loadImagesFromFolder(base_folder=IMAGE_DIR ,debug=False, loadSIFT=False,only_images=True,type="Normals")
        normals = stitcher.stitch_images_rgb(images_grid)
        # save normals
        output_name = get_unique_output_name(os.path.join(IMAGE_DIR, os.path.basename(IMAGE_DIR)), "normals", "tiff")
        cv2.imwrite(output_name, normals)
        print(f"---[SAVING IMAGE {output_name}]---")
        
    if is_stitch_reflections:
        print(f"---[MERGING REFLECTIONS IMAGE]---")
        images_grid=loadImagesFromFolder(base_folder=IMAGE_DIR ,debug=False, loadSIFT=False,only_images=True,type="ReflectionMap")
        reflectionMap = stitcher.stitch_images_rgb(images_grid)
        output_name = get_unique_output_name(os.path.join(IMAGE_DIR, os.path.basename(IMAGE_DIR)), "reflections", "tiff")
        cv2.imwrite(output_name, reflectionMap)
        print(f"---[SAVING IMAGE {output_name}]---")

    if is_stitch_mesh:
        print(f"---[MERGING MESH]---")
        # save global homography into file that can be loaded for mesh alignment
        mf= MeshFusion(IMAGE_DIR)
        mf.load_transform_matrices(image_height=image_size[1])
        mf.apply_transform_matrices()
        mf.global_alignment()
        mf.crop_border(dilation_steps=25, save_cleaned=False)
        mf.save_project(mesh_name="mesh.ply")
        mf.save_transformation_matrices()
        tm, indexes=mf.get_transformation_matrices_with_indices()
        # extract names using indexes
        #print("--- preparing cam folders for texturing ---")
        print(f"---[PREPARING CAM FOLDERS FOR TEXTURING]---")
        # folder names (A1,A2...) organize a folder for texturing containing image and image.cam files 
        folder_names = load_grid_names(IMAGE_DIR)
        folder_names = [folder_names[r][c] for (r,c) in indexes]
        # get one of the image sizes
        print("--- Saving albedo cam folder for texturing")
        prepare_cam_folder(IMAGE_DIR, copy.deepcopy(gh), indexes, folder_names,type="albedo",pano_size=final_pano_size)
        print("--- Saving normals cam folder for texturing")
        prepare_cam_folder(IMAGE_DIR, copy.deepcopy(gh), indexes, folder_names, type="Normals",pano_size=final_pano_size)
        print("--- Saving reflection cam folder for texturing")
        prepare_cam_folder(IMAGE_DIR, copy.deepcopy(gh), indexes, folder_names, type="ReflectionMap",pano_size=final_pano_size)
