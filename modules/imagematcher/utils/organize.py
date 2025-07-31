import os
import re
import shutil
import argparse

def organize_tif_images(base_path):
    """
    Organizes .tif images into folders based on their grid identifiers.
    :param base_path: Path to the directory containing .tif images.
    """
    # Regex pattern to match grid identifiers (e.g., A1, B2, etc.)
    pattern = re.compile(r"^([A-Za-z]\d+).*\.tif$", re.IGNORECASE)

    # Iterate over all files in the base path
    for filename in os.listdir(base_path):
        # Check if the file is a .tif file
        if filename.lower().endswith(".tif"):
            # Match the grid identifier
            match = pattern.match(filename)
            if match:
                grid_id = match.group(1)  # Extract the grid identifier (e.g., A1, B2, etc.)
                # Create the folder path
                folder_path = os.path.join(base_path, grid_id)
                # Create the folder if it doesn't exist
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                # Move the file to the folder
                src_path = os.path.join(base_path, filename)
                dest_path = os.path.join(folder_path, "albedo.tif")
                shutil.move(src_path, dest_path)
                print(f"Moved {filename} to {folder_path}")
            else:
                print(f"Skipping {filename}: No grid identifier found.")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Organize .tif images into folders based on grid identifiers.")
    parser.add_argument("path", type=str, help="Path to the directory containing .tif images.")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if the provided path exists
    if not os.path.exists(args.path):
        print(f"Error: The path '{args.path}' does not exist.")
        return
    
    # Organize .tif images
    organize_tif_images(args.path)

if __name__ == "__main__":
    main()