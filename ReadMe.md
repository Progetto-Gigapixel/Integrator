# Gigapixel Integrator

A modular system for high-resolution image processing and integration.

## Prerequisites

### System Requirements

- Python (Python 3.10 required)
- venv module (included with Python)
- RawTherapee 5.11 (for SHAFT module)
- EXIFTOOL
- git
- cmake
- .NET Framework 4.6.2 and [Edge Runtime](https://developer.microsoft.com/en-us/microsoft-edge/webview2/) installed
- [MSYS2 and MinGW toolchain] (for ImageMatcher texturing)

## Installation

### 1. SHAFT Module Setup

For the SHAFT module to work correctly, set the RAWTHERAPEE_PATH in .env with the installation path
[Rawtherapee windows](https://rawtherapee.com/downloads/5.11/)
https://rawtherapee.com/downloads/5.11/

Create `.env` file with RawTherapee path:

```bash
#example
RAWTHERAPEE_PATH = C:\Program Files\RawTherapee\5.11
```

Install the EXIFTOOL following the instructions

- https://exiftool.org/

### 2. Imagematcher texturing

Install MSYS2 from <https://www.msys2.org/>

Copy mingw32-make with name make in the same directory

After installation, update the package database and core packages:
Run these commands in the MSYS2 MinGW ucr terminal:
https://www.msys2.org/docs/environments/

```bash
pacman -Syu
pacman -S mingw-w64-ucrt-x86_64-gcc
# Install compiler toolchain
# Required libraries
pacman -U https://repo.msys2.org/mingw/ucrt64/mingw-w64-ucrt-x86_64-libtiff-4.6.0-1-any.pkg.tar.zst
pacman -U https://repo.msys2.org/mingw/ucrt64/mingw-w64-ucrt-x86_64-libjpeg-turbo-3.0.1-1-any.pkg.tar.zst
pacman -U https://repo.msys2.org/mingw/ucrt64/mingw-w64-ucrt-x86_64-intel-tbb-1~2020.3-2-any.pkg.tar.zst
pacman -U https://repo.msys2.org/mingw/ucrt64/mingw-w64-ucrt-x86_64-libpng-1.6.49-1-any.pkg.tar.zst
pacman -U https://repo.msys2.org/mingw/ucrt64/mingw-w64-ucrt-x86_64-zlib-1.3.1-1-any.pkg.tar.zst
```

build the mvs texuting library

```bash
cd .\\modules\\imagematcher\\mvs-texturing
cmake -G "MinGW Makefiles" -S . -B build -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_MAKE_PROGRAM="C:/msys64/ucrt64/bin/mingw32-make" -DTBB_ROOT_DIR="C:/msys64/ucrt64" -DCMAKE_PREFIX_PATH="C:/msys64/ucrt64"
cd build
make

```

### 3. webview2 Run-time

Install the edge webview runtime for Windows
https://developer.microsoft.com/en-us/microsoft-edge/webview2/

### Configuration

The system uses config.json for configuration. Default values are set during installation:

```json
{
  "app": {
    "MAX_CONTENT_HEIGHT_MM": 375,
    "MAX_CONTENT_WIDTH_MM": 500,
    "APP_PORT": 5000,
    "SECRET_KEY": "your-secret-key-here",
    "CAPTURER_URL": "http://127.0.0.1:8081",
    "RAWTHERAPEE_PATH": "...",
    "PHOCUS_EXE_PATH": "...",
    "UNITY_EXE_PATH": "..."
  },
  "paths": {
    "CAPTURER_DIR": "...",
    "SHAFT_DIR": "...",
    "NLIGHTS_DIR": "...",
    "IMAGEMATCHER_DIR": "..."
  },
  "venvs": {
    "PYTHON_PATH": "...",
    "SHAFT_VENV_PYTHON_PATH": "...",
    "NLIGHTS_VENV_PYTHON_PATH": "...",
    "IMAGEMATCHER_VENV_PYTHON_PATH": "..."
  },
  "imagematcher": {
    "IMAGEMATCHER_TEXTURING_PATH": "...",
    "IMAGEMATCHER_TEXTURING_SCRIPT": "...",
    "IMAGEMATCHER_TEXTURING_EXECUTABLE": "..."
  }
}
```

if you already have a path and venv for each modules you can override them using `.env` file before running the installation

```env
PYTHON_PATH=...
SHAFT_VENV_PYTHON_PATH=...
SHAFT_DIR=...
NLIGHTS_VENV_PYTHON_PATH=...
NLIGHTS_DIR=...
IMAGEMATCHER_DIR=...
IMAGEMATCHER_VENV_PYTHON_PATH=...
```

### Installation script

```bash
# activate the venv
python -m venv envs\venv
envs\venv\Script\activate
python setup.py
```

## Execution

By python script

```bash
python app.py
```

By executable

```bash
./Integrator
```

## Structure

```md
gigapixel-integrator/
├── envs/ # Virtual environments
├── modules/ # Module directories
│ ├── capturer/ # Capturer module
│ ├── shaft/ # Shaft module
│ ├── nlights/ # NLights module
│ └── imagematcher/ # ImageMatcher module
├── static/ # folder has all css and js files
├── templates/ # contain all jinja2 html templates for pages
├── config.json # Configuration file
├── requirements.txt # Base requirements
└── setup.py # Setup script
```

- **static** folder has all css and js files

- **templates** contain all jinja2 html templates for pages

- **modules** has all the standalone application that the integrator must call under the hood
