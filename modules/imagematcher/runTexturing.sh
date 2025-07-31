#!/bin/bash

# Check if arguments are provided
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <texrecon_path> <test_images_path> <folder_name>"
    echo "Example: $0 mvs-texturing/build/apps/texrecon/texrecon.exe test_images my_folder"
    exit 1
fi

TEXRECON_PATH="$1"
TEST_IMAGES_PATH="$2"
FOLDER_NAME="$3"

# Check if folder name is provided
if [ -z "$FOLDER_NAME" ]; then
    echo "Error: Please provide a folder name as the third argument."
    exit 1
fi

# Run texrecon commands
"$TEXRECON_PATH" "$TEST_IMAGES_PATH/$FOLDER_NAME/texture_albedo" "$TEST_IMAGES_PATH/$FOLDER_NAME/mesh.ply" "$TEST_IMAGES_PATH/$FOLDER_NAME/tex/tex$FOLDER_NAME"
"$TEXRECON_PATH" "$TEST_IMAGES_PATH/$FOLDER_NAME/texture_Normals" "$TEST_IMAGES_PATH/$FOLDER_NAME/mesh.ply" "$TEST_IMAGES_PATH/$FOLDER_NAME/tex/tex$FOLDER_NAME""Normals" -L "$TEST_IMAGES_PATH/$FOLDER_NAME/tex/tex$FOLDER_NAME""_labeling.vec" -D "$TEST_IMAGES_PATH/$FOLDER_NAME/tex/tex$FOLDER_NAME""_data_costs.spt"
"$TEXRECON_PATH" "$TEST_IMAGES_PATH/$FOLDER_NAME/texture_ReflectionMap" "$TEST_IMAGES_PATH/$FOLDER_NAME/mesh.ply" "$TEST_IMAGES_PATH/$FOLDER_NAME/tex/tex$FOLDER_NAME""Reflection" -L "$TEST_IMAGES_PATH/$FOLDER_NAME/tex/tex$FOLDER_NAME""_labeling.vec" -D "$TEST_IMAGES_PATH/$FOLDER_NAME/tex/tex$FOLDER_NAME""_data_costs.spt"

# --- Step 1: Delete unwanted files ---
echo "Deleting non-TIFF Reflection/Normal files in $TEST_IMAGES_PATH/$FOLDER_NAME/tex/..."
find "$TEST_IMAGES_PATH/$FOLDER_NAME/tex/" \( -iname "*Reflection*" -o -iname "*Normal*" \) -not -iname "*.tiff" -exec rm -v {} \;

# --- Step 2: Update .mtl file ---
MTL_FILE="$TEST_IMAGES_PATH/$FOLDER_NAME/tex/tex$FOLDER_NAME.mtl"
if [ ! -f "$MTL_FILE" ]; then
    echo "Error: .mtl file not found at $MTL_FILE"
    exit 1
fi

echo "Updating $MTL_FILE..."
TEMP_FILE="${MTL_FILE}.tmp"

# Variables to track current material and existing lines
current_material=""
has_map_Ks=""
has_map_Kn=""

while IFS= read -r line; do
    # Detect new material section
    if [[ "${line:0:6}" == "newmtl" ]]; then
        current_material="$line"
        has_map_Ks=""
        has_map_Kn=""
    fi
    
    # Check for existing map_Ks/map_Kn in current material
    if [ -n "$current_material" ]; then
        if [[ "${line:0:6}" == "map_Ks" ]]; then has_map_Ks=1; fi
        if [[ "${line:0:6}" == "map_Kn" ]]; then has_map_Kn=1; fi
    fi
    
    # Replace Ks values
    line="${line/Ks 0.000000 0.000000 0.000000/Ks 1.000000 1.000000 1.000000}"
    echo "$line"
    
    # Process map_Kd lines
    if [[ "$line" == *"map_Kd"* ]]; then
        filename="${line#map_Kd }"
        IFS="_" read -r prefix rest <<< "$filename"
        
        # Add map_Ks if not already present in this material
        if [ -z "$has_map_Ks" ]; then
            echo "map_Ks ${prefix}Reflection_${rest}_map_Kd.tiff"
        fi
        
        # Add map_Kn if not already present in this material
        if [ -z "$has_map_Kn" ]; then
            echo "map_Kn ${prefix}Normal_${rest}_map_Kd.tiff"
        fi
    fi
done < "$MTL_FILE" > "$TEMP_FILE"

mv -f "$TEMP_FILE" "$MTL_FILE"
echo ".mtl file updated successfully."  