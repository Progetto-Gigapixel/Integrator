from flask import Blueprint, jsonify,render_template, session, current_app
import requests 
from enum import Enum
import subprocess
import json
import os
capturer_api = Blueprint('capturer_api', __name__, url_prefix='/api/capturer')


# generate a config.json for the capturer
def generate_capturer_config(path):
    config={
        "step_x": session.get('stepx',245)/10, #convert to cm
        "step_y": session.get('stepy',185)/10, #convert to cm
        "threshold": 3.8,   
        "delta": session.get('vibration_check_time',3)
    }
    sensitivity=session.get('sensitivity','low')

    if sensitivity == 'low':
        config['threshold'] = 3.5
    capturer_json_file_path=os.path.join(path, 'config.json')
    #save the config to a json file
    # create the file if it doesn't exist
    if not os.path.exists(capturer_json_file_path):
        open(capturer_json_file_path, 'w').close()
    with open(capturer_json_file_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Config file generated at {os.path.join(path, 'config.json')}")
    return True


def letter_to_number(letter):
    return ord(letter.lower()) - ord('a') + 1

def number_to_letter(number):
    if 1 <= number <= 26:
        return str(chr(ord('A') + number - 1))  # Use 'A' for uppercase letters
    
def call_external_api(api_url, payload=None):
    try:
        if payload:
            response = requests.post(api_url, json=payload, timeout=60)
        else:
            response = requests.get(api_url, timeout=60)
        
        response.raise_for_status()  # Raises exception for 4XX/5XX errors
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": str(e)}
class Command(Enum):
    STEP = 'step'
    FREE = 'free'
    STOP = 'stop'

class Direction(Enum):
    LEFT = 'a'
    RIGHT = 'd'
    UP = 'w'
    DOWN = 's'

lights = {i: False for i in range(1, 11)}

@capturer_api.route('/move/<command>/<direction>')
def move(command, direction):   
    CAPTURER_URL=current_app.config['CAPTURER_URL'] 
    try:
        cmd = Command(command)
        dir = Direction(direction)
    except ValueError:
        return "Invalid command or direction", 400
    external_response = call_external_api(CAPTURER_URL+f"/move/{cmd.value}/{dir.value}")
    return jsonify(external_response),200

@capturer_api.route('/lights/<int:light_id>/<status>')
def single_light_control(light_id, status):
    CAPTURER_URL=current_app.config['CAPTURER_URL'] 
    if light_id not in lights:
        return jsonify({"status": "error", "message": "Invalid light ID"}), 400
    
    if status == 'on':
        lights[light_id] = True
    elif status == 'off':
        lights[light_id] = False
    else:
        return jsonify({"status": "error", "message": "Invalid status"}), 400
    #Ritorna un json con lo stato delle luci
    print(CAPTURER_URL+f"/lights/{light_id}/{status}")
    external_response = call_external_api(CAPTURER_URL+f"/lights/{light_id}/{status}")
    print(external_response)
    return jsonify(external_response),200


@capturer_api.route('/lights/<command>')
def bulk_light_control(command):
    CAPTURER_URL=current_app.config['CAPTURER_URL'] 
    if command not in ['allOn', 'allOff', '1234on']:
        return jsonify({"status": "error", "message": "Invalid command"}), 400
    external_response = call_external_api(CAPTURER_URL+f"/lights/{command}")
    if command == 'allOff':
        for light_id in lights:
            lights[light_id] = False
        return jsonify({"status": "success", 
                        "message": "All lights turned off", 
                        "lights": {k: v for k, v in lights.items() if k in [1,2,3,4]},
                        "external_response": external_response})
    elif command == '1234on':
        for light_id in [1, 2, 3, 4]:
            lights[light_id] = True
        return jsonify(external_response),200
    
    return jsonify({"status": "error", "message": "Invalid command"}), 400



@capturer_api.route('/checkMovements')
def check_movements():
    CAPTURER_URL=current_app.config['CAPTURER_URL'] 
    external_response = call_external_api(CAPTURER_URL+f"/checkMovements")
    if external_response['status'] == 'error':
        print(external_response)
        return jsonify(external_response),500
    return jsonify(external_response),200


@capturer_api.route('/phocus')
def phocus():   
    PHOCUS_EXE_PATH=current_app.config['PHOCUS_EXE_PATH'] 
    try:
        #run phocus as process
        subprocess.Popen([PHOCUS_EXE_PATH], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

    return jsonify({"status": "ok", "message": "Phocus opened"}), 200