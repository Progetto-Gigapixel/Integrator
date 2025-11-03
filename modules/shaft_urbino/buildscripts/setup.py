import os
import subprocess
import sys
from cx_Freeze import setup, Executable

def check_venv():
    if os.getenv("VIRTUAL_ENV") is None:
        print("Error: You are not inside a virtual environment (venv).")
        sys.exit(1)

def check_directory():
    current_dir = os.path.basename(os.getcwd())
    print(f"Current directory: {current_dir}")
    if current_dir != "buildscripts":
        print(f"Error: The current directory is '{current_dir}', but it should be 'buildscripts'.")
        sys.exit(1)

def install_requires():
    requirements_path = os.path.join(os.path.dirname(__file__), "..", "requirements.txt")
    requirements_path = os.path.abspath(requirements_path)
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", requirements_path], check=True)

def fix_opencv_libs():
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "opencv-python", "opencv-python-headless"],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "opencv-python-headless"],
        check=True,
    )

def build_exe():
    build_exe_options = {
        "packages": ["os", "cv2"],  # aggiungi qui i tuoi moduli principali
        "includes": [],
        "excludes": [],
        "include_files": []  # file extra da includere
    }

    setup(
        name="CO.CO",
        version="1.0",
        description="CO.CO.A - Color Correction App",
        options={"build_exe": build_exe_options},
        executables=[Executable("../main.py", target_name="coco.exe")]
    )

if __name__ == "__main__":
    check_directory()
    check_venv()
    install_requires()
    fix_opencv_libs()

    # Avvia la build
    build_exe()
