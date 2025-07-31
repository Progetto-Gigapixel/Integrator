import configparser
import json
import os
import pathlib
import sys
from pathlib import Path

import lensfunpy
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
# from PyQt5.QtWidgets import QApplication


def get_target_reference(campioni_colore, patch_number):
    return np.array(
        campioni_colore[str(patch_number)]
    )  # Get the target white color (normalized to 0-1 range)


# Function to start the application
def start_app(mainWindow):
    app = QApplication(sys.argv)
    window = mainWindow()
    window.show()
    sys.exit(app.exec_())


# Read an image (RGB)
def read_image(file_path):
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(file_path) as img:
        img = img.convert("RGB")
        img_array = np.array(img)
    return img_array


# Function to display a spacer
def spacer():
    print("\n" + "-" * 50 + "\n")



def find_project_root(markers=( "config.ini", "main.py")):
    """
    Risale la gerarchia a partire da current_path finché trova uno dei marker.
    Restituisce la base directory del progetto o None se non trovata.
    """
    current_file = Path(__file__).resolve()
    current_path = current_file.parent
    for parent in [current_path] + list(current_path.parents):
        for marker in markers:
            if (parent / marker).exists():
                return parent
    return None

# Function to read the configuration file
def read_config():
    config = configparser.ConfigParser()
    directory_path = find_project_root()
    ini_path = os.path.join(directory_path, "config.ini")
    config.read(ini_path)
    return config


def get_config(parent: str, child: str):
    config = read_config()

    return config.get(parent, child)


def get_nikon_lens_name(lensID):
    config = read_config()
    nikon_tags = os.path.abspath(config.get("directories","tags_path"), f"nikon_tags.json")

    with open(nikon_tags, "r") as f:
        data = json.load(f)

    normalized_lensID = str(lensID).strip().upper().replace(" ", "")

    for tag in data:
        normalized_tag_id = tag["id"].strip().upper().replace(" ", "")
        if normalized_tag_id == normalized_lensID:
            return tag["name"]

    return None


# Get the lensfun database adding the XML custom files
def get_lensfun_db():
    config = read_config()
    dir_path = Path(config.get("directories", "lensfun_path"))
    db_files = [str(file) for file in dir_path.rglob("*.xml")]

    db = lensfunpy.Database(db_files)
    return db


def show_image(image, window_name="Image"):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title(window_name)
    plt.show()



BRADFORD = np.array(
    [[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]]
)
VON_KRIES = np.array(
    [
        [0.40024, 0.70760, -0.08081],
        [-0.22630, 1.16532, 0.04570],
        [0.00000, 0.00000, 0.91822],
    ]
)
SHARP = np.array(
    [[1.2694, -0.0988, -0.1706], [-0.8364, 1.8006, 0.0357], [0.0297, -0.0315, 1.0018]]
)
CAT2000 = np.array(
    [[0.7982, 0.3389, -0.1371], [-0.5918, 1.5512, 0.0406], [0.0008, 0.2390, 0.9753]]
)
CAT02 = np.array(
    [[0.7328, 0.4296, -0.1624], [-0.7036, 1.6975, 0.0061], [0.0030, 0.0136, 0.9834]]
)




def update_measured_values_in_patches(patches, measured_values):
    idx = 0
    for rgb_value in measured_values:
        patches[0][idx]["rgb_values"] = rgb_value
        idx += 1
    return patches

# Before saving to json, in case there're non serializable objects
def convert_numpy(obj):
    if isinstance(obj, np.integer):  # Per numeri interi NumPy
        return int(obj)
    if isinstance(obj, np.floating):  # Per numeri float NumPy
        return float(obj)
    if isinstance(obj, np.ndarray):  # Per array NumPy
        return obj.tolist()
    raise TypeError(f"Type {type(obj)} not serializable")


# Convert numpy arrays to lists
def convert_numpy_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_numpy_to_list(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_to_list(i) for i in obj]
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj
