import os
import re
import argparse


parser = argparse.ArgumentParser(description="Organize .tif images into folders based on grid identifiers.")
parser.add_argument("path", type=str, help="Path to the directory containing .tif images.")

# Parse arguments
args = parser.parse_args()

if not os.path.exists(args.path):
    print(f"Error: The path '{args.path}' does not exist.")
    exit()
    
# Define the base directory containing the folders
base_path = args.path
# Get all folder names in the directory
folders = sorted([f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))])

# Extract unique letter prefixes (A-Z)
letters = sorted(set(re.match(r"([A-Z])\d+", f).group(1) for f in folders if re.match(r"([A-Z])\d+", f)), reverse=True)

# Create mapping from original letters to reversed letters
mapping = {original: new for original, new in zip(sorted(letters), letters)}

# First Pass: Rename folders to temporary names
temp_mapping = {}
for folder in folders:
    match = re.match(r"([A-Z])(\d+)", folder)
    if match:
        original_letter, number = match.groups()
        new_letter = mapping.get(original_letter, original_letter)  # Get mapped letter
        temp_name = f"temp_{new_letter}{number}"  # Temporary name

        old_path = os.path.join(base_path, folder)
        temp_path = os.path.join(base_path, temp_name)

        os.rename(old_path, temp_path)
        temp_mapping[temp_name] = f"{new_letter}{number}"  # Store final name

# Second Pass: Rename temporary folders to final names
for temp_name, final_name in temp_mapping.items():
    os.rename(os.path.join(base_path, temp_name), os.path.join(base_path, final_name))
    print(f"Renamed: {temp_name} --> {final_name}")