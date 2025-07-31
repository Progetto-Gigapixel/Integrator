
from flask import Blueprint,session,current_app,Response
import subprocess
import os 
import re
import sys
shaft_route = Blueprint('shaft_route', __name__,url_prefix='/api/shaft')


def colorchecker_results_handler():
    return

@shaft_route.route('/colorchecker')
def colorchecker():
    script_dir = current_app.config['SHAFT_DIR']
    script_name = 'main.py'
    python_path = current_app.config['SHAFT_VENV_PYTHON_PATH']
    
    command = [
        python_path,
        script_name,
        "-i", session.get("shaft_colorchecker_path", "/"),
        "-w", session.get("shaft_flatfield_file_path", "/"),
        "-c", session.get("shaft_output_colorspace", "srgb"),
        "-o", session.get("shaft_savein_path", "/"),
        "-f", "tif",
        "-m", "AM",
    ]
    
    if session.get("shaft_light_balance", False):
        command.append("-lb")
    if session.get("shaft_sharpen", False): 
        command.append("-s")
        
    output_path = session.get("shaft_savein_path", "/")
    log_path = os.path.join(output_path, "log/session_logs.log")

    def stream_stdout():
        process = None
        try:
            # Create process with appropriate platform-specific settings
            kwargs = {
                'cwd': script_dir,
                'stdout': subprocess.PIPE,
                'stderr': subprocess.STDOUT,
                'text': True,
                'bufsize': 1,
            }
            if sys.platform == "win32":
                kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW

            process = subprocess.Popen(command, **kwargs)
            
            progress = 0
            color_correction = 0
            pattern = re.compile(r'.*INFO.*')  
            poly_pattern = re.compile(r'.*polynomial_correction:.*')

            for line in iter(process.stdout.readline, ''):
                if not line:  # End of stream
                    break
                    
                line = line.strip()
                print(line)
                
                # Check if client has disconnected
                try:
                    # This will raise a GeneratorExit if the client disconnected
                    # We'll use a dummy yield to test the connection
                    yield f"data: {progress}\n\n"
                except GeneratorExit:
                    print("Client disconnected - terminating process")
                    if process.poll() is None:
                        process.terminate()
                    raise
                
                # Process the line as before
                if pattern.search(line):
                    progress += 1
                    yield f"data: {progress}\n\n"
                    
                if poly_pattern.search(line):
                    print(line)
                    color_correction = float(line.split("polynomial_correction:")[1])
                    color_correction = round(color_correction, 6)
                    yield f"data: Color correction : {color_correction}\n\n"

            # Process completed normally
            process.wait()
            if process.returncode != 0:
                error_msg = f"Process failed with return code {process.returncode}"
                print(error_msg)
                yield f"data: ERROR: {error_msg}\n\n"
            
            yield "data: done\n\n"

        except GeneratorExit:
            # Client disconnected - clean up
            print("Client disconnected during processing")
            if process and process.poll() is None:
                print("Terminating running process...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
            raise
            
        except Exception as error:
            print(f"Error in process: {str(error)}")
            yield f"data: ERROR: {str(error)}\n\n"
            yield "data: PROCESS_FAILED\n\n"
            yield "data: done\n\n"
            
        finally:
            # Clean up resources
            if process:
                if process.poll() is None:
                    process.terminate()
                process.stdout.close()
                if process.stderr:
                    process.stderr.close()

    return Response(stream_stdout(), mimetype='text/event-stream')



@shaft_route.route('/development/stream')
def development_stream():
    python_path = current_app.config['SHAFT_VENV_PYTHON_PATH']
    script_dir = current_app.config['SHAFT_DIR']
    script_name = 'main.py'
    colorcheckerFilePath = session.get("shaft_colorchecker_path", "/")
    saveinPath = session.get("shaft_savein_path", "/")
    colorcheckerFileName = os.path.splitext(os.path.basename(colorcheckerFilePath))[0]
    profilePath = os.path.join(saveinPath, "params/"+colorcheckerFileName+"correction_params.json")
    development_folder_path = session.get("shaft_develop_folder_path", "/")
    output_path = session.get("shaft_output_path", "/")
    process_format = session.get("shaft_process_format", "tif")
    
    # Count files to process
    num_files = len([f for f in os.listdir(development_folder_path) 
                   if f.lower().endswith(process_format.lower())])
    print(f"Found {num_files} files to process")
    
    command = [
        python_path,
        script_name,
        "-i", development_folder_path,
        "-o", output_path,
        "-f", "tif",
        "--parameter-path", profilePath,
        "--extension", process_format,
        "-m", "DM",
    ]
    if session.get("shaft_process_subfolders", False):
        command.append("--process-subfolder")
    if session.get("shaft_overwrite", False):
        command.append("--overwrite-files")
    
    print("Command:", " ".join(command))

    def stream_stdout():
        process = None
        try:
            # Create process with appropriate settings
            kwargs = {
                'cwd': script_dir,
                'stdout': subprocess.PIPE,
                'stderr': subprocess.STDOUT,
                'text': True,
                'bufsize': 1,
            }
            if sys.platform == "win32":
                kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW

            process = subprocess.Popen(command, **kwargs)
            print("Process started")
            
            progress = 0
            progress_step = 100 / (num_files + 1) if num_files > 0 else 100
            pattern = re.compile(r'.*EXIF metadata transferred successfully.*')
            
            for line in iter(process.stdout.readline, ''):
                if not line:  # End of stream
                    break
                    
                line = line.strip()
                print(line)
                
                # Check if client has disconnected
                try:
                    # Test connection with dummy yield
                    yield f"data: {progress:.2f} \n\n"
                except GeneratorExit:
                    print("Client disconnected - terminating process")
                    if process.poll() is None:
                        process.terminate()
                    raise
                
                # Process the line
                if pattern.search(line):
                    progress = min(progress + progress_step, 100)  # Ensure we don't exceed 100%
                    yield f"data: {progress:.2f}\n\n"
            
            # Process completed
            process.wait()
            print("Process completed")
            
            if process.returncode != 0:
                error_msg = f"Process failed with return code {process.returncode}"
                print(error_msg)
                yield f"data: ERROR: {error_msg}\n\n"
            
            yield "data: done\n\n"

        except GeneratorExit:
            # Handle client disconnect
            print("Client disconnected during processing")
            if process and process.poll() is None:
                print("Terminating running process...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
            raise
            
        except Exception as error:
            print(f"Error in process: {str(error)}")
            yield f"data: ERROR: {str(error)}\n\n"
            yield "data: PROCESS_FAILED\n\n"
            yield "data: done\n\n"
            
        finally:
            # Clean up resources
            if process:
                if process.poll() is None:
                    process.terminate()
                process.stdout.close()
                if process.stderr:
                    process.stderr.close()

    return Response(stream_stdout(), mimetype='text/event-stream')