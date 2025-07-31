
from flask import Blueprint,session, current_app,Response,jsonify
import subprocess
import os 
import re
import sys
from pathlib import Path
imagematcher_route = Blueprint('imagematcher_route', __name__,url_prefix='/api/imagematcher')

def remove_first_ld_library_path():
    ld_path = os.getenv("LD_LIBRARY_PATH")
    if ld_path:
        paths = ld_path.split(":")
        if len(paths) > 1:
            new_path = ":".join(paths[1:])  # Skip first entry
        return new_path
    return None

@imagematcher_route.route('/alignment/stream')
def alignment_stream():
    python_path = current_app.config['IMAGEMATCHER_VENV_PYTHON_PATH']
    script_dir = current_app.config['IMAGEMATCHER_DIR']
    script_name = 'GUI/main.py'
    input_path=session.get("imagematcher_maps_path")
    command = [
        python_path,
        script_name, 
        input_path,
          "--m", "au",
          "--prv"
    ]
    myenv = os.environ.copy()
    #venv_path=os.path.join(current_app.config['IMAGEMATCHER_VENV_PYTHON_PATH'], "envs","imagematcher")
    #myenv['LD_LIBRARY_PATH'] = remove_first_ld_library_path() 
    print(command)
    def stream_stdout():
        try:

            process = subprocess.Popen(
                command,
                cwd=script_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=myenv
            )
            print("start")
            progress = 0

            errorPattern = re.compile(r'.*Errno.*')
            pattern = re.compile(r'.*Matching progress*')  
            for line in iter(process.stdout.readline, ''):
                print(">>> "+line)
                line = line.strip()
                match = pattern.search(line)
                if match:
                    #extract progress ({progress}%)
                    # print(line)
                    progress = float(line.split("(")[1].split("%")[0])
                    #parse to int 
                    progress = int(progress)
                    print(progress)
                    yield f"data: {progress}\n\n" 

            process.wait()
            process.stdout.close()

            yield "data: done\n\n"  # Final message
        except Exception as error:
            print(f"Error in process: {str(error)}")
            yield f"data: ERROR: {str(error)}\n\n"
            yield "data: PROCESS_FAILED\n\n"
            yield "data: done\n\n"
    return Response(stream_stdout(), mimetype='text/event-stream')

@imagematcher_route.route('/alignment/manual')
def manual_alignment():
    python_path = current_app.config['IMAGEMATCHER_VENV_PYTHON_PATH']
    script_dir = current_app.config['IMAGEMATCHER_DIR']
    script_name = 'GUI/main.py'

    input_path=session.get("imagematcher_maps_path")
    command = [
        python_path,
        script_name, 
        input_path,
          "--m", "ma",
    ]
    process= subprocess.run(command, cwd=script_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(process)
    return jsonify({"status": "ok", "message": "GUI opened"}), 200


@imagematcher_route.route('/merge/stream')
def merge_stream():
    python_path = current_app.config['IMAGEMATCHER_VENV_PYTHON_PATH']
    script_dir = current_app.config['IMAGEMATCHER_DIR']
    script_name = 'GUI/main.py'

    input_path=session.get("imagematcher_maps_path")
    command = [
        python_path,
        script_name, 
        input_path,
          "--m", "fi","--sn","--sr","--sm"
    ]
    #TEXTURING 
    tex_path = current_app.config['IMAGEMATCHER_TEXTURING_PATH']
    tex_executable = current_app.config['IMAGEMATCHER_TEXTURING_EXECUTABLE']
    tex_script = current_app.config['IMAGEMATCHER_TEXTURING_SCRIPT']
    tex_input_path=session.get("imagematcher_maps_path")
    tex_output_path=session.get("imagematcher_tex_out_path")
    tex_command = [
        tex_script, 
        tex_input_path, tex_output_path, tex_executable
    ]
    def stream_stdout():
        try:
            process = subprocess.Popen(
                command,
                cwd=script_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            print("start")
            progress = 0
            errorPattern = re.compile(r'.*Errno.*')
            pattern = re.compile(r'.*Matching progress*')  
            for line in iter(process.stdout.readline, ''):
                line = line.strip()
                print(line)
                match = pattern.search(line)
                if match:
                    #extract progress ({progress}%)
                    # print(line)
                    progress = float(line.split("(")[1].split("%")[0])
                    #parse to int 
                    progress = int(progress)
                    # print(progress)
                    yield f"data: {progress}\n\n" 

            process.wait()
            process.stdout.close()
            #TEXTURING
            if not os.path.exists(os.path.join(tex_output_path, "tex")):
                os.makedirs(os.path.join( tex_output_path, "tex"))
            print("start tex")
            print(tex_command)
            process= subprocess.run(tex_command, cwd=tex_path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            # print("done")
            # Read and check stderr after process completion
            yield "data: done\n\n"  # Final message
        except Exception as error:
            print(f"Error in process: {str(error)}")
            yield f"data: ERROR: {str(error)}\n\n"
            yield "data: PROCESS_FAILED\n\n"
            yield "data: done\n\n"
    return Response(stream_stdout(), mimetype='text/event-stream')


