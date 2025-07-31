
from flask import Blueprint,session,Response, current_app, jsonify
import subprocess
import datetime
import os 
import zipfile
import shutil

modelviewer_route = Blueprint('modelviewer_route', __name__,url_prefix='/api/modelviewer')

def generate_directory_name(name, author):
    now = datetime.datetime.now()
    components = [
        str(now.year),
        f"{now.month:02d}",
        f"{now.day:02d}",
        f"{now.hour:02d}",
        f"{now.minute:02d}",
        f"{now.second:02d}",
        name,
        author
    ]
    # Join with underscores and replace spaces
    return "_".join(components).replace(" ", "_")
def create_info(file_path, name, author, width, length):
    info_file = os.path.join(file_path, "info.txt")
    file_content = ""
    file_content += f"{name}\n"
    file_content += f"{author}\n"
    file_content += f"{length}\n"
    file_content += f"{width}\n"
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write(file_content)
def extract_zip_efficiently(zip_path, new_directory_path):
    """
    Efficiently extract zip file without loading all content into memory
    """
    try:
        # Ensure destination directory exists
        os.makedirs(new_directory_path, exist_ok=True)
        print(f"Extracting template to: {new_directory_path}")        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            total_files = len(file_list)
            print(f"Total files in zip: {total_files}")
            for i, file_info in enumerate(file_list, 1):
                try:
                    # Extract individual file
                    zip_ref.extract(file_info, new_directory_path)
                    # Calculate progress
                    percentage = int((i / total_files) * 100)
                    yield f"data: {percentage}\n\n"
                    print(f"Extracted {file_info} ({percentage}% complete)")
                except Exception as file_error:
                    print(f"Error extracting {file_info}: {file_error}")
        #Return success
        print(f"Successfully extracted template to: {new_directory_path}") 
        yield "data: error\n\n"
    except Exception as e:
        print(f"Error extracting zip: {e}")
        yield "data: error\n\n"
@modelviewer_route.route('/unity/template')
def templating():
    """
    This function takes 2 paths from the session and copies the content of the template
    to the output directory. It also generates a new directory name based on the current
    time and the name/author of the artwork. It also creates an info.txt file with the
    name and author of the artwork. Finally it runs a command in the target directory.
    """
    files=[]
    # load from the session 2 path 
    templateDirPath = session.get("modelviewer-template-dir-path", "/")
    artworkDirPath = session.get("modelviewer-artwork-dir-path", "/")
    #TESTING PURPOSE
    templateDirPath="/home/hjcsteve/gigapixel/Integrator/test/mv/templates"
    artworkDirPath="/home/hjcsteve/gigapixel/Integrator/test/mv/artwork"
    name= "test"
    author="testAuthor"
    zip_name = "one-shot.zip"
    # check if the template dir exists
    if not os.path.exists(templateDirPath):
        return {"stdout": "Template directory does not exist", "stderr": "Template directory does not exist","returncode": 1}, 500
    # check if the output dir exists
    if not os.path.exists(artworkDirPath):
        return {"stdout": "Output directory does not exist", "stderr": "Output directory does not exist","returncode": 1}, 500
    #GENERATE THE NEW DIRECTORY NAME
    directory_name = generate_directory_name(name, author)
    new_directory_path = os.path.join(os.curdir, directory_name)
    os.mkdir(new_directory_path)
    # GENERATE the INFO FILE
    create_info(new_directory_path, name, author, 1, 1)
    print(f'new_directory_path: {new_directory_path}')
    zip_path = os.path.join(templateDirPath, zip_name)
    extract_zip_efficiently(zip_path, new_directory_path)
    print(f'new_directory_path: {new_directory_path}')
    # copies the artwork to the new directory under Assets/Artwork/Maps
    maps_dir = os.path.join(new_directory_path, "Assets/Artwork/Maps/")
    print(f'maps_dir: {maps_dir}')
    if not os.path.exists(maps_dir):
        os.makedirs(maps_dir)
    # copy all files and subdirectories
    for root, dirs, files in os.walk(artworkDirPath):
        for file in files:
            shutil.copy(os.path.join(root, file), os.path.join(maps_dir, file))
        for dir in dirs:
            shutil.copytree(os.path.join(root, dir), os.path.join(maps_dir, dir))
    print(f"Successfully copied artwork to: {artworkDirPath}")
    # Run the command in the target directory
    json_result = {"stdout": "Success", "stderr": "Success","returncode": 0}
    return json_result, 200


@modelviewer_route.route('/unity/template/stream')
def templating_stream():
    """
    This function takes 2 paths from the session and copies the content of the template
    to the output directory. It also generates a new directory name based on the current
    time and the name/author of the artwork. It also creates an info.txt file with the
    name and author of the artwork. Finally it runs a command in the target directory.
    """
    files=[]
    # load from the session 2 path 
    templateDirPath = session.get("modelviewer_template_dir_path", "/")
    artworkDirPath = session.get("modelviewer_artwork_dir_path", "/")
    name= session.get("title", "")
    author=session.get("artist", "")    
    zip_name = session.get("modelviewer_template_zip", "")
    outputDirPath = session.get("modelviewer_output_path", "/")


    # check if the template dir exists
    if not os.path.exists(templateDirPath):
        return {"stdout": "Template directory does not exist", "stderr": "Template directory does not exist","returncode": 1}, 500
    # check if the output dir exists
    if not os.path.exists(artworkDirPath):
        return {"stdout": "Output directory does not exist", "stderr": "Output directory does not exist","returncode": 1}, 500
    def stream_stdout():
        #GENERATE THE NEW DIRECTORY NAME
        directory_name = generate_directory_name(name, author)
        print
        new_directory_path = os.path.join(outputDirPath, directory_name)
        os.mkdir(new_directory_path)
        # GENERATE the INFO FILE
        create_info(new_directory_path, name, author, 1, 1)
        print(f'new_directory_path: {new_directory_path}')
        zip_path = os.path.join(templateDirPath, zip_name)
        # extract_zip_efficiently(zip_path, new_directory_path)
        print(f'zip_path: {zip_path}')
        try:
            # Ensure destination directory exists
            os.makedirs(new_directory_path, exist_ok=True)
            print(f"Extracting template to: {new_directory_path}")
            # print the progress
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                total_files = len(file_list)
                print(f"Total files in zip: {total_files}")
                for i, file_info in enumerate(file_list, 1):
                    try:
                        # Extract individual file
                        zip_ref.extract(file_info, new_directory_path)
                        # Calculate progress
                        percentage = int((i / total_files) * 100)
                        yield f"data: {percentage}\n\n"
                        print(f"Extracted {file_info} ({percentage}% complete)")
                    except Exception as file_error:
                        print(f"Error extracting {file_info}: {file_error}")
                        yield "data: error\n\n"
                #Return success
                print(f"Successfully extracted template to: {new_directory_path}")      
        except Exception as e:
            print(f"Error extracting template: {e}")
            yield "data: error\n\n"
        print(f'new_directory_path: {new_directory_path}')
        # copies the artwork to the new directory under Assets/Artwork/Maps
        maps_dir = os.path.join(new_directory_path, "Assets/Artwork/Maps/")
        print(f'maps_dir: {maps_dir}')
        if not os.path.exists(maps_dir):
            os.makedirs(maps_dir)
        # copy all files and subdirectories
        for root, dirs, files in os.walk(artworkDirPath):
            for file in files:
                shutil.copy(os.path.join(root, file), os.path.join(maps_dir, file))
            for dir in dirs:
                shutil.copytree(os.path.join(root, dir), os.path.join(maps_dir, dir))
        print(f"Successfully copied artwork to: {artworkDirPath}")
        # json_result = {"stdout": "Success", "stderr": "Success","returncode": 0}
        yield "data: done\n\n" 
    return Response(stream_stdout(), mimetype='text/event-stream')


@modelviewer_route.route('/unity/open')
def phocus():   
    UNITY_EXE_PATH=current_app.config['UNITY_EXE_PATH'] 
    try:
        #run phocus as process
        subprocess.Popen([UNITY_EXE_PATH], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

    return jsonify({"status": "ok", "message": "UNITY opened"}), 200