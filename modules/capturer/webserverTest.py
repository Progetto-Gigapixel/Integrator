from flask import Flask, jsonify
import socket
import time
import json



app = Flask(__name__)

translate_light_status = {True:"ON",False:"OFF"}
step_translation={
    "a":"j",
    "s":"k",
    "d":"l",
    "w":"i"
}

def letter_to_number(letter):
    return ord(letter.lower()) - ord('a') + 1

def number_to_letter(number):
    if 1 <= number <= 26:
        return str(chr(ord('A') + number - 1))  # Use 'A' for uppercase letters
    
class Stativo:

    def __init__(self, stepx, stepy):
        self.max_horizontal = 144.  
        self.max_vertical = 126.

        self.stepx = stepx
        self.stepy = stepy

        self.max_steps_vertical = self.max_vertical // stepy
        self.max_steps_horizontal = self.max_horizontal // stepx

        self.ledStatus = [True]*8
        self.vibrations = False
        self.position = [1,1]
        self.is_moving = False

    def get_state(self):
        return {
            "move":{
                "state": self.is_moving,
                "position": number_to_letter(self.position[0])+str(self.position[1])
            },
            "lights": self.ledStatus,
            "vibratios": self.vibrations
        } 





@app.route('/move/<string:action>/<string:direction>', methods=['GET'])
def move(action, direction):
    if action not in ['step', 'free', 'stop']:
        return jsonify({'error': 'Invalid action'}), 400
    
    if action in ['step','free']:
        if direction not in ['d','a','s','w']:
            return jsonify({'error': 'Invalid direction'}), 400

    if action == 'step':
        # Qui aspetta di arrivare alla posizione, perché s.recv è bloccante.
        # Quindi ritornerà sempre un valore di vibrazioni falso.

        if direction == 'w' and struct.position[0] >= struct.max_steps_vertical:
            return jsonify(struct.get_state())

        if direction == 's' and struct.position[0] <= 1:
            return jsonify(struct.get_state())
        
        if direction == 'd' and struct.position[1] >= struct.max_steps_horizontal:
            return jsonify(struct.get_state())
        
        if direction == 'a' and struct.position[1] <= 1:
            return jsonify(struct.get_state())
        

        # ret = s.send((step_translation[direction]).encode())
        # val = s.recv(1)
        # struct.vibrations = bool(int.from_bytes(val, "big"))
        struct.vibrations = False
        print(f"Step done, vibrations: {struct.vibrations}")
        struct.is_moving = False


        # Manca solo il controllo, preferibilmente prima della comunicazione, sullo stato
        if direction == 'w':
            struct.position[0]+=1
        elif direction == 's':
            struct.position[0]-=1
        elif direction == 'd':
            struct.position[1]+=1
        elif direction == 'a':
            struct.position[1]-=1

    elif action == 'free':
        ret = s.send(direction.encode())
        print(f"Free movements, dir:{direction}")
        struct.is_moving = True
    
    else:
        ret = s.send(b'0')
        print("Stopped")
        struct.is_moving = False

    return jsonify(struct.get_state())

@app.route('/lights/<string:light_id>/<string:state>', methods=['GET'])
def control_light(light_id, state):
    light_id = int(light_id)
    if state not in ['on', 'off']:
        return jsonify({'error': 'Invalid state'}), 400

    state = True if state == 'on' else False

    if struct.ledStatus[light_id-1] != state:
        ret = s.send(str(light_id).encode())
        struct.ledStatus[light_id-1] = state
    print(struct)
    return jsonify(struct.get_state())


@app.route('/lights/<string:command>', methods=['GET'])
def control_lights(command):
    if command == 'allOff':
        for i in range(8):
                if struct.ledStatus[i]:
                    ret = s.send(str(i+1).encode())
                    struct.ledStatus[i] = False
                    time.sleep(0.3)


    elif command.endswith('on'):
        for i in range(4):
                if not struct.ledStatus[i]:
                    ret = s.send(str(i+1).encode())
                    time.sleep(0.3)
                    struct.ledStatus[i] = True

                if struct.ledStatus[7-i]: 
                    ret = s.send(str(8-i).encode())
                    time.sleep(0.3)
                    struct.ledStatus[7-i] = False


    else:
        return jsonify({'error': 'Invalid command'}), 400
    
    return jsonify(struct.get_state())



@app.route('/checkMovements', methods=['GET'])
def check_movements(): 
    ret = s.send("c".encode())
    val = s.recv(1)
    # val = bool(int.from_bytes(val, "big"))
    # struct.vibrations = bool(int.from_bytes(val, "big"))
    # print(val)
    val = bool(int.from_bytes(val, "big"))
    # print(val)
    struct.vibrations = not val
    print(f"Step done, vibrations: {struct.vibrations}")
    return jsonify(struct.get_state())


if __name__ == '__main__':

    with open('config.json','r') as f:
        cfg = json.load(f)


    global s
    global struct 
    struct=Stativo(stepx=cfg['step_x'], stepy=cfg['step_y'])

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # s.connect(("192.48.56.2",80))
    # print("Opening socket")
    # #s.connect(("127.0.0.1",8082))
    # time.sleep(2) 
    # print(s)

    # # Sending configuration
    # s.send(f"x{cfg['step_x']}y{cfg['step_y']}d{cfg['delta']}t{cfg['threshold']}\n".encode())

    print("Sended configuration ", cfg)

    print(f"Max steps number: h {struct.max_steps_horizontal}, v {struct.max_steps_vertical} ")
    app.run(debug=True, port=8081, use_reloader=False)
