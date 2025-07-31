import os
import sys
from dotenv import load_dotenv
from utility.configManager import ConfigManager
# Take environment variables from .env

if hasattr(sys, '_MEIPASS'):
    basedir = os.getcwd()
else:
    #current dir
    basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'), override=True)
json_config = ConfigManager(os.path.join(basedir, 'config.json'))
class Config:
    BASEDIR = basedir
    SECRET_KEY =os.getenv('SECRET_KEY') or json_config.get_value("app",'SECRET_KEY') or 'your-secret-key-here'
    SQLALCHEMY_DATABASE_URI =os.getenv('DATABASE_URL') or json_config.get_value("app",'DATABASE_URL') or \
        'sqlite:///' + os.path.join(basedir, 'app.db')
    APP_PORT = os.getenv('APP_PORT') or json_config.get_value("app",'APP_PORT') or 5000
    CAPTURER_URL = os.getenv('CAPTURER_URL') or json_config.get_value("app",'CAPTURER_URL') or "http://127.0.0.1:8081"
    MAX_CONTENT_HEIGHT_MM =os.getenv('MAX_CONTENT_HEIGHT_MM') or json_config.get_value("app",'MAX_CONTENT_HEIGHT_MM') or 375
    MAX_CONTENT_WIDTH_MM =os.getenv('MAX_CONTENT_WIDTH_MM') or json_config.get_value("app",'MAX_CONTENT_WIDTH_MM') or 500
    RAWTHERAPEE_PATH =os.getenv('RAWTHERAPEE_PATH') or json_config.get_value("app","RAWTHERAPEE_PATH") or "C:\\Program Files\\RawTherapee\\5.11"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    PHOCUS_EXE_PATH =os.getenv('PHOCUS_EXE_PATH') or json_config.get_value("app","PHOCUS_EXE_PATH") or './test.sh'
    UNITY_EXE_PATH =os.getenv('UNITY_EXE_PATH') or json_config.get_value("app","UNITY_EXE_PATH") or './test.sh'

    # from json config
    PYTHON_PATH = os.getenv('PYTHON_PATH') or json_config.get_value("venvs","PYTHON_PATH") or 'python3'
    CAPTURER_DIR = os.getenv('CAPTURER_DIR') or json_config.get_value("paths","CAPTURER_DIR") or 'modules/capturer'
    SHAFT_DIR = os.getenv('SHAFT_DIR') or json_config.get_value("paths",'SHAFT_DIR') or "modules/shaft"
    SHAFT_VENV_PYTHON_PATH = os.getenv('SHAFT_VENV_PYTHON_PATH') or json_config.get_value("venvs","SHAFT_VENV_PYTHON_PATH")  or 'envs/shaft/bin/python3'
    NLIGHTS_DIR = os.getenv('NLIGHTS_DIR') or json_config.get_value("paths",'NLIGHTS_DIR') or "modules/nlights"
    NLIGHTS_VENV_PYTHON_PATH = os.getenv('NLIGHTS_VENV_PYTHON_PATH') or json_config.get_value("venvs","NLIGHTS_VENV_PYTHON_PATH")  or 'envs/nlights/bin/python3'
    IMAGEMATCHER_DIR = os.getenv('IMAGEMATCHER_DIR') or json_config.get_value("paths",'IMAGEMATCHER_DIR') or "modules/imagematcher"
    IMAGEMATCHER_VENV_PYTHON_PATH = os.getenv('IMAGEMATCHER_VENV_PYTHON_PATH') or json_config.get_value("venvs","IMAGEMATCHER_VENV_PYTHON_PATH")  or 'envs/imagematcher/bin/python3'


    IMAGEMATCHER_TEXTURING_PATH = os.getenv('IMAGEMATCHER_TEXTURING_PATH') or json_config.get_value("imagematcher",'IMAGEMATCHER_TEXTURING_PATH') or '/path/to/venv'
    IMAGEMATCHER_TEXTURING_EXECUTABLE = os.getenv('IMAGEMATCHER_TEXTURING_EXECUTABLE') or json_config.get_value("imagematcher",'IMAGEMATCHER_TEXTURING_EXECUTABLE') or '/path/to/venv'
    IMAGEMATCHER_TEXTURING_SCRIPT = os.getenv('IMAGEMATCHER_TEXTURING_SCRIPT') or json_config.get_value("imagematcher",'IMAGEMATCHER_TEXTURING_SCRIPT') or '/path/to/venv'