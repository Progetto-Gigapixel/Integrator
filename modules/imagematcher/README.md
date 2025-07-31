# GigaStitch

## Requirements

The `<image-directory>` must be structured according to a grid format (rows x cols).
The first level of subdirectories represents the rows in the grid. These folders must be named in alphabetic order.
Inside each row folder, the images represent the columns in the grid. These images must be named in numeric order.

Example Directory Structure

```txt
<image-directory>/
├── A1/
│   ├── *albedo.tiff
│   ├── *ReflectionMap.tiff
│   ├── *Normals.tiff
│   └── *.ply
|
├── A2/
│   ├── *albedo.tiff
│   ├── *ReflectionMap.tiff
│   ├── *Normals.tiff
│   └── *.ply
├── B1/
│   ├── *albedo.tiff
│   ├── *ReflectionMap.tiff
│   ├── *Normals.tiff
│   └── *.ply
...
├── C2/
│   ├── *albedo.tiff
│   ├── *ReflectionMap.tiff
│   ├── *Normals.tiff
│   └── *.ply
```

## Requirement installation

```bash
pip install -r requirements.txt
```

# Execution

## Stitching module

How to execute python modules for image stitching and mesh fusion

```bash
python3 main.py --<options> <image-directory>
```

1. Automatic image stitching for grayscale images (au mode)
2. Manual GUI (ma mode)
3. Finalized image stitching for albedo,Reflection maps, Normals and Meshes (fi mode)

example usage

```bash
python3 GUI/main.py --md all low
```

```bash

usage: main.py [-h] [--m {au,ma,fi,all}] [--prv] [--ba] [--not_ba] [--snf SNF] [--sol SOL] [--sct SCT] [--set SET] [--ss SS]
               [--md MD] [--mi MI] [--rt RT] [--lsf LSF] [--lmf LMF] [--sr] [--sn] [--sm] [--bnp BNP]
               input

positional arguments:
  input               Input file

options:
  -h, --help          show this help message and exit
  --m {au,ma,fi,all}  Mode of operation: au (automatic), ma (manual), fi (final), all (automatic and final)
  --prv               Stitch preview
  --ba                Use bundle adjustment
  --not_ba            Do not use bundle adjustment
  --snf SNF           Number of features to use in SIFT
  --sol SOL           Number of octave layers to use in SIFT
  --sct SCT           Contrast threshold to use in SIFT
  --set SET           Edge threshold to use in SIFT
  --ss SS             Sigma to use in SIFT
  --md MD             Match distance to use in matching
  --mi MI             Minimum inliers to use in matching
  --rt RT             RANSAC threshold to use in matching
  --lsf LSF           Path to load SIFT from file
  --lmf LMF           Path to load matches from file
  --sr                Stitch reflections
  --sn                Stitch normals
  --sm                Stitch mesh
  --bnp BNP           Numbers of bundle parameters to optimize for each pair of images
```

## Texturing module

### Build with cmake

build the executable

```bash
cd mvs-texturing
mkdir build
cmake -S . -B build
cd build
make
```

### (Windows cmd) cmd

```bash
runTexturing.bat "path\to\texrecon.exe" "path\to\test_images" "folder_name"
```
