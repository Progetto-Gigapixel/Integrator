import os
import dotenv
import sys
import platform
from subprocess import call
from utility.configManager import ConfigManager
import PyInstaller.__main__
# Load environment variables
dotenv.load_dotenv()
basedir = os.path.abspath(os.path.dirname(__file__))
config= ConfigManager()

def run_command(command):
    """Helper function to run OS-specific commands"""
    if platform.system() == "Windows":
        call(command, shell=True)
    else:
        call(command, shell=True, executable="/bin/bash")

def create_venv(df=None):
    """Create virtual environment and install requirements"""
    venv_path = os.path.join(basedir, "envs", "venv")
    
    # Create envs directory if it doesn't exist
    os.makedirs("envs", exist_ok=True)

    # Create venv
    run_command(f"python -m venv {venv_path}")
    
    # Install requirements
    if platform.system() == "Windows":
        activate_cmd = f"{venv_path}\\Scripts\\activate.bat && "
    else:
        activate_cmd = f"source {venv_path}/bin/activate && "
    
    run_command(f"{activate_cmd}python -m pip install --upgrade pip")
    run_command(f"{activate_cmd}pip install -r requirements.txt")
    #config.update_value("venvs","PYTHON_PATH",venv_path+"/bin/python")
    if platform.system() == "Windows":
        config.update_value("venvs","PYTHON_PATH",venv_path+"\\Scripts\\python.exe")
    else:
        config.update_value("venvs","PYTHON_PATH",venv_path+"/bin/python")



def check_venv():
    """Check if running inside a virtual environment."""
    if not hasattr(sys, 'real_prefix') and sys.prefix == sys.base_prefix:
        print("Error: You are not inside a virtual environment (venv).")
        sys.exit(1)


def load_module_directory(req_dir_var,df=None):
    req_dir = os.getenv(req_dir_var) or df
    if not req_dir:
        print(f"Error: Environment variable {req_dir_var} not set!")
        sys.exit(1)
    # check if dir exists
    if not os.path.exists(req_dir):
        print(f"Error: Environment variable {req_dir_var} does not exist!")
        sys.exit(1)
    # update config with DIR value
    config.update_value("paths",req_dir_var,req_dir)
    print(f"Using {req_dir} as {req_dir_var} directory")
    return req_dir
def create_env(env_name, req_dir_var,df=None):
    """Generic function to create any environment"""
    req_dir=load_module_directory(req_dir_var,df)
    venv_path = os.getenv(env_name+"_VENV_PYTHON_PATH")
    if not venv_path:
        venv_path = os.path.join(basedir,"envs", env_name.lower())
    else:
        print(f"Using {venv_path} as venv path")
        config.update_value("venvs",env_name+"_VENV_PYTHON_PATH",venv_path)
        return True
    
    # Clean up existing venv if it exists
    if os.path.exists(venv_path):
        import shutil
        shutil.rmtree(venv_path, ignore_errors=True)
    
    os.makedirs("envs", exist_ok=True)
    
    # Use specific Python executable if needed
    #python_exe = "python3.10" if env_name.lower() == "shaft" else "python"
    python_exe="python.exe"
    try:
        run_command(f"{python_exe} -m venv {venv_path}")
        
        if platform.system() == "Windows":
            activate_cmd = f"{venv_path}\\Scripts\\activate.bat && "
        else:
            activate_cmd = f"source {venv_path}/bin/activate && "
        
        run_command(f"{activate_cmd} python -m pip install --upgrade pip setuptools wheel")
        
        # Install requirements one by one for better error handling
        requirements_file = os.path.join(req_dir, "requirements.txt")
        if os.path.exists(requirements_file):
            run_command(f"{activate_cmd}pip install -r {requirements_file} --verbose")
        else:
            print(f"Warning: Requirements file not found at {requirements_file}")
            
    except Exception as e:
        print(f"Failed to create {env_name} environment: {str(e)}")
        if os.path.exists(venv_path):
            shutil.rmtree(venv_path, ignore_errors=True)
        sys.exit(1)
    # update config json with VENV value
    #config.update_value("venvs",env_name+"_VENV_PYTHON_PATH",venv_path+"/bin/"+python_exe)
    if platform.system() == "Windows":
        config.update_value("venvs",env_name+"_VENV_PYTHON_PATH",venv_path+"\\Scripts\\python.exe")
    else:
        config.update_value("venvs",env_name+"_VENV_PYTHON_PATH",venv_path+"/bin/python")
    return True

# Execute functions
try:
    create_venv()
    print("Base venv created")
    capturer_dir = load_module_directory("CAPTURER_DIR", os.path.join(basedir,"modules", "capturer"))
    create_env("SHAFT", "SHAFT_DIR", os.path.join(basedir,"modules", "shaft"))
    print("Shaft venv created")
    create_env("NLIGHTS", "NLIGHTS_DIR", os.path.join(basedir,"modules", "nlights"))
    print("Nlights venv created")
    create_env("IMAGEMATCHER", "IMAGEMATCHER_DIR", os.path.join(basedir,"modules", "imagematcher"))
    print("Imagematcher venv created")
    check_venv()
    print("All modules installed")
    # generate config.json with default values
    config.update_value("app","MAX_CONTENT_HEIGHT_MM",375)
    config.update_value("app","MAX_CONTENT_WIDTH_MM",500)
    config.update_value("app","APP_PORT",5000)
    config.update_value("app","SECRET_KEY","your-secret-key-here")
    config.update_value ("app","CAPTURER_URL","http://127.0.0.1:8081")
    config.update_value ("app","RAWTHERAPEE_PATH","C:\\Program Files\\RawTherapee\\5.11")
    config.update_value ("app","PHOCUS_EXE_PATH","")
    config.update_value ("app","UNITY_EXE_PATH","")
    #config.json with texturing
    config.update_value("imagematcher","IMAGEMATCHER_TEXTURING_PATH", os.path.join(basedir,"modules", "imagematcher","mvs-texturing"))
    config.update_value("imagematcher","IMAGEMATCHER_TEXTURING_SCRIPT",os.path.join(basedir,"modules", "imagematcher","runTexturing.sh"))
    config.update_value("imagematcher","IMAGEMATCHER_TEXTURING_EXECUTABLE",os.path.join(basedir,"modules", "imagematcher","mvs-texturing","build","apps","texrecon","texrecon"))
    
    
    ##installing with pyinstaller
    PyInstaller.__main__.run([
    'app.py',
    '--onefile',
    '--add-data=templates:templates',
    '--add-data=static:static',
    '--distpath=.',
    "--clean",
    "--name",
    "Integrator",
    "--noconfirm",
    "--noconsole",
    "--icon=.\\static\\icon.ico"
    ])

except Exception as e:
    print(f"Error: {str(e)}")
    sys.exit(1)