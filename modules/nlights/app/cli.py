"""
Main entry point for the Photometric Stereo CLI application 
"""
import sys
import os

import click
import configparser

# Crea l'oggetto ConfigParser
configIni = configparser.ConfigParser()



# Set up path before imports
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

#from PyQt5.QtWidgets import QApplication,QFileDialog,QMessageBox,QWidget
from core.normal_maps import compute_normal_maps_new    

def setup_environment():
    """Configure environment variables and paths"""
    # Configure default paths
    app_data_dir = os.path.join(base_dir, 'data')
    if not os.path.exists(app_data_dir):
        os.makedirs(app_data_dir)


# Leggi il file config.ini
config_path = os.path.join(os.path.dirname(__file__), 'config.ini')
configIni.read(config_path)


light_strenght_file_path = configIni['Assets']['light_strenght_file_path']

lights_file_path_def_v = configIni['Assets']['lights_file_path_def_v']
black_error_v = configIni['Assets']['black_error_v']
black_error_2_v = configIni['Assets']['black_error_2_v']

lights_file_path_def_h = configIni['Assets']['lights_file_path_def_h']
black_error_h = configIni['Assets']['black_error_h']
black_error_2_h = configIni['Assets']['black_error_2_h']

lights_file_path_cust_v = configIni['Assets']['lights_file_path_cust_v']
lights_file_path_cust_h = configIni['Assets']['lights_file_path_cust_h']

pymeshlab_virtual_environment_absolut_path = configIni['Paths']['pymeshlab_virtual_environment_absolut_path']



@click.group()
def groupcli():
    pass


setup_environment()

config = {
    "lights_file_path": "",
    "black_error": "",
    "black_error_2": "",
    "light_strenght_file_path": light_strenght_file_path,
    "loadOptions":None,
    'output_directory': '',
    'all_lights_on_image': '',
    'DDLightsOrder':'45N...15W',
    'StrelSize': 5,
    'light_direction_images': [],
    'height_map_format': 'PLY',
    'decimation_method': 'Accurate',
    'decimation_surface': '1',
    'poly_correction': True,
    'image_downsample': False,
    'normal_map': None,
    'depth_map': None,
    'albedo': None
    }


    

def setup(self,setup, default:bool=True):
    """Handle param in setup command"""
    config["pymeshlab_virtual_environment_absolut_path"] = pymeshlab_virtual_environment_absolut_path
    if default:
        if setup == "Vertical":
            config["lights_file_path"] = lights_file_path_def_v #"./assets/Default_Vertical.txt"
            config["black_error"] = black_error_v #"./assets/Black_Error_Vertical.tif"
            config["black_error_2"] = black_error_2_v #"./assets/Black_Error_Vertical_2.tif"
        else:
            config["lights_file_path"] = lights_file_path_def_h #"./assets/Default_Horizontal.txt"
            config["black_error"] = black_error_h #"./assets/Black_Error_Horizontal.tif"
            config["black_error_2"] = black_error_2_h #"./assets/Black_Error_Horizontal_2.tif"
    else:
        if setup == "Vertical":
            config["lights_file_path"] = lights_file_path_cust_v #"./assets/Custom_Vertical.txt"
            config["black_error"] = black_error_v #"./assets/Black_Error_Vertical.tif
            config["black_error_2"] = black_error_2_v #"./assets/Black_Error_Vertical_2.tif"
        else:
            config["lights_file_path"] = lights_file_path_cust_h #"./assets/Custom_Horizontal.txt"
            config["black_error"] = black_error_h #"./assets/Black_Error_Horizontal.tif"
            config["black_error_2"] = black_error_2_h #"./assets/Black_Error_Horizontal_2.tif"   
        
        '''
        light_pos_path, _  = QFileDialog.getOpenFileName(self,
           "Select custom file for light positions", "", "Files (*.txt)")
        #QFileDialog.getExistingDirectory(self, "Select Input Image Directory")
        if light_pos_path:
            config["lights_file_path"] = light_pos_path  
        else:
            click.echo("No custom light positions file selected. Using default settings.")
            '''
    click.echo(f"Setup changed to: {setup}")
    click.echo(f"Using default: {default}")


def select_input_allLights(central_widget,param):
    """Open a dialog to select the input directory containing images"""

    if param=='al':
        image_path, _  = QFileDialog.getOpenFileName(parent=central_widget,
            caption="Select All Lights Image", directory="", filter="Image Files (*.jpg *.jpeg *.png *.tif *.tiff)")

        if image_path:
            config["all_lights_on_image"] = image_path
            click.echo(image_path)
        else:
            click.echo("No image selected for all lights. Please select an image.")

    else:
        if param=='ld':
            file_paths, _ = QFileDialog.getOpenFileNames(parent=central_widget,
            caption="Select Light Direction Images", directory="", filter="Image Files (*.jpg *.jpeg *.png *.tif *.tiff)")
        
        if file_paths:
            config['light_direction_images'] = file_paths
            click.echo(file_paths)
        else:
            click.echo("No images selected for light directions. Please select images.")        


def select_output_directory(self):
    """Open a dialog to select the output directory"""
    directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
    if directory:
        config['output_directory'] = directory
        click.echo(f"Output directory set to: {directory}")
    else:
        click.echo("No directory selected.")



def configs(light_order, height_map_format, decimation_method, decimation_surface, poly_correction, image_downsample):
    """Handle param in config command"""
    config['DDLightsOrder'] = light_order
    config['height_map_format'] = height_map_format
    config['decimation_method'] = decimation_method
    config['decimation_surface'] = decimation_surface
    config['poly_correction'] = poly_correction
    config['image_downsample'] = image_downsample
    

#POSSIBLE COMMAND: process 
@groupcli.command()
#@click.option('--set',"-s", default='Horizontal', prompt="Inserisci lo stativo: Vertical/Horizontal", type=str)
#@click.option('--defa',"-d", default=True, prompt="Scegli asset di default o custom", type=bool)
#@click.option('--light_order', "-lo", default='45N...15W', prompt="Inserisci l'ordine delle luci: 45W... 15S/45N... 15W", type=str)
#@click.option('--height_map_format', "-hmf", default='PLY', prompt="Inserisci il formato della mappa di profondità: PLY/STL", type=str)
#@click.option('--decimation_method', "-dm", default='Accurate', prompt="Inserisci il metodo di decimazione: Accurate/Coarse", type=str)
#@click.option('--decimation_surface', "-ds", default='1', prompt="Inserisci la superficie di decimazione: 1/2/3/...", type=str)
#@click.option('--poly_correction', "-pc", default=True, prompt="Attiva correzione poligonale: True/False", type=bool)  
#@click.option('--image_downsample', "-id", default=True, prompt="Attiva downsample immagine: True/False", type=bool)
@click.option('--set',"-s", default='Horizontal', help="Inserisci lo stativo: Vertical/Horizontal", type=str)
@click.option('--defa',"-d", default=True, help="Scegli asset di default o custom", type=bool)
@click.option('--light_order', "-lo", default='45N...15W', help="Inserisci l'ordine delle luci: 45W...15S/45N...15W", type=str)
@click.option('--height_map_format', "-hmf", default='PLY', help="Inserisci il formato della mappa di profondità: PLY/STL", type=str)
@click.option('--decimation_method', "-dm", default='Accurate', help="Inserisci il metodo di decimazione: Accurate/Coarse", type=str)
@click.option('--decimation_surface', "-ds", default='1', help="Inserisci la superficie di decimazione: 1/2/3/...", type=str)
@click.option('--poly_correction', "-pc", default=True, help="Attiva correzione poligonale: True/False", type=bool)  
@click.option('--image_downsample', "-id", default=False, help="Attiva downsample immagine: True/False", type=bool)
@click.option('--all_lights_image', "-al", required=True, help="Inserisci il path dell'immagine con tutte le luci", type=str)
@click.option('--direction_images', "-ld", required=True, help="Inserisci i path delle immagini con luci direzionali separati da virgola senza spazi", type=str)
@click.option('--output_directory', "-od", required=True, help="Inserisci il path della directory di output", type=str)
@click.option('--help', "-h", is_flag=True, required=False, type=bool)
def process(set, defa, light_order, height_map_format, decimation_method, decimation_surface, poly_correction, image_downsample, all_lights_image,direction_images, output_directory, help):
    
    if help:
        help_mess()
    else:    
        # appQ = QApplication(sys.argv)
        # central_widget = QWidget()
        setup(None,set, defa)
        configs(light_order, height_map_format, decimation_method, decimation_surface, poly_correction, image_downsample)

        config['output_directory'] = output_directory
        config["all_lights_on_image"] = all_lights_image
        config['light_direction_images'] = direction_images.split(',') if direction_images else []


        '''
        click.echo('Seleziona immagine con tutte le luci')
        select_input_allLights(central_widget,param='al')
        click.echo('Seleziona immagini con le luci direzionali')
        select_input_allLights(central_widget,param='ld') 
        click.echo('Seleziona la cartella di output')
        select_output_directory(central_widget)   
        '''   

        config['loadOptions'] ={
            'ImageChannel': 1,
            'NormalizePercentile': 99, #IN MATLAB è STATICO!!!
            'resample': config['image_downsample'] #PRENDI DAL CHECKBOX!!
        }

        results = compute_normal_maps_new(config)




def help_mess():
    print(
    "Available commands:\n"
    #"  process - Process images with photometric stereo. Without options, follow the prompts to set up the environment and select images.\n"
    #"  (example with options) process -s Vertical -d True -lo 45W...15S -hmf PLY -dm Accurate -ds 1 -pc True -id True\n"
    "  process -> Options:\n"
    "     -s, --set TEXT (Vertical/Horizontal)\n"
    "     -d, --defa BOOLEAN\n"
    "     -lo, --light_order TEXT (45W...15S/45N...15W)\n"
    "     -hmf, --height_map_format TEXT (PLY/STL)\n"
    "     -dm, --decimation_method TEXT (Accurate/Coarse)\n"
    "     -ds, --decimation_surface TEXT (int from 1 to ...)\n"
    "     -pc, --poly_correction BOOLEAN\n"
    "     -id, --image_downsample BOOLEAN\n"
    "     -al, --all_lights_image TEXT (Path to image with all lights)\n"
    "     -ld, --direction_images TEXT (Paths to direction images separated by commas without spaces)\n"
    "     -od, --output_directory TEXT (Path to output directory)\n"
    "     --help    Show this message and exit.\n\n"
    "  help - Show this help message and exit.\n"
    )





@groupcli.command()
#@click.option('--help', "-h",type=str)
def help():
    """Show help message"""

    click.echo(
    "Available commands:\n"
    #"  process - Process images with photometric stereo. Without options, follow the prompts to set up the environment and select images.\n"
    #"  (example with options) process -s Vertical -d True -lo 45W...15S -hmf PLY -dm Accurate -ds 1 -pc True -id True\n"
    "  process -> Options:\n"
    "     -s, --set TEXT (Vertical/Horizontal)\n"
    "     -d, --defa BOOLEAN\n"
    "     -lo, --light_order TEXT (45W...15S/45N...15W)\n"
    "     -hmf, --height_map_format TEXT (PLY/STL)\n"
    "     -dm, --decimation_method TEXT (Accurate/Coarse)\n"
    "     -ds, --decimation_surface TEXT (int from 1 to ...)\n"
    "     -pc, --poly_correction BOOLEAN\n"
    "     -id, --image_downsample BOOLEAN\n"
    "     -al, --all_lights_image TEXT (Path to image with all lights)\n"
    "     -ld, --direction_images TEXT (Paths to direction images separated by commas without spaces)\n"
    "     -od, --output_directory TEXT (Path to output directory)\n"
    "     --help    Show this message and exit.\n\n"
    "  help - Show this help message"
    )



if __name__ == "__main__":
    groupcli()