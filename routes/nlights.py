
from flask import Blueprint,session,current_app,Response
import subprocess
import os 
import re
from pathlib import Path

nlights_route = Blueprint('nlights_route', __name__,url_prefix='/api/nlights')


def colorchecker_results_handler():
    return


@nlights_route.route('/compute/stream')
def computate_maps_stream():
    python_path = current_app.config['NLIGHTS_VENV_PYTHON_PATH']
    script_dir = current_app.config['NLIGHTS_DIR']
    script_name = 'app/cli.py'
    setupType = session.get("stand_type","Vertical")
    #the first letter to capitalize
    setupType = setupType[0].upper() + setupType[1:]
    output_path=session.get("nlights_output_directory", "/")
    nlights_grid_folder = session.get("nlights_grid_folder", "A1")
    final_output_path=os.path.join(output_path,nlights_grid_folder)
    if not os.path.exists(final_output_path):
        os.makedirs(final_output_path)
    print(final_output_path)
    nlights_all_lights_image_dir = session.get("nlights_all_lights_image_dir", "/")
    nlights_direction_images_dir = session.get("nlights_direction_images_dir", "")

    command = [
        python_path,
        script_name,
        "process",
        "-al", nlights_all_lights_image_dir,
        "--set", setupType,
        "--output_directory", final_output_path,
        '-id','false'
    ]
    dir_array = nlights_direction_images_dir.split(",")
    windows_paths = [str(Path(dir_path.strip()).resolve()) for dir_path in dir_array]
    command_arg = f'{",".join(windows_paths)}'
    command.append("-ld")
    command.append(command_arg) 

    command_string = ' '.join(f'"{arg}"' if ' ' in arg or '\\' in arg else arg for arg in command)
    print(command_arg)
    print(command_string)
    def stream_stdout():
        try:
            # if len(dir_array)==8:
            #     command.append("-ld")
            #     command.append(nlights_direction_images_dir)
            # else:
            #     print("no direction images")
            #     yield "data: done\n\n"
            #     raise Exception("no direction images")
            
            process = subprocess.Popen(
                command,
                cwd=script_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            print("start")
            progress = 0
            progress_step= 100/5
            # print(progress_step)
            pattern = re.compile(r'.*Qualcosa*')  
            for line in iter(process.stdout.readline,''):
                line = line.strip()
                match = pattern.search(line)
                print(line)
                if match:
                    # print(line)
                    progress = progress + progress_step
                    yield f"data: {progress}\n\n" 
            process.stdout.close()
            process.wait()
            print("done")
               # Read and check stderr after process completion
            stderr_output = process.stderr.read()
            process.stderr.close()
            
            if stderr_output.strip():
                #get last line
                lines = stderr_output.splitlines()
                last_line = lines[-1]
                raise Exception(f"Process stderr: {stderr_output}",last_line)
            yield "data: done\n\n"  # Final message
            
        except Exception as error:
            print(f"Error in process: {str(error)}")
            yield f"data: ERROR: {str(error)}\n\n"
            yield "data: PROCESS_FAILED\n\n"
            yield "data: done\n\n"
    return Response(stream_stdout(), mimetype='text/event-stream')