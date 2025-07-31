import sys

from PyQt5.QtWidgets import (
    QApplication, QDialog, QMainWindow, QMessageBox
)
from PyQt5.uic import loadUi

from mainWindow import Ui_MainWindow
from stepWind import Ui_StepWindow
import serial
import time
import socket
import time




translation = {
    1 : '1',
    2 : '2',
    3 : '3',
    4 : '4',
    5 : '5',
    6 : '6',
    7 : '7',
    8 : '8',
    9 : 'a',
    10 : 'd',
    11 : 'w',
    12 : 's',
    13 : "off",
    14 : "on",
    15 : "stop",
    16 : "l",
    17 : "j",
    18 : "i",
    19 : "k",
    20 : "c"
   
}

ledStatus = [True]*8



translate_light_status = {True:"ON",False:"OFF"}






class Window(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.button1.clicked.connect(lambda checked, n=1: self.light_handler(n))
        self.button2.clicked.connect(lambda checked, n=2: self.light_handler(n))
        self.button3.clicked.connect(lambda checked, n=3: self.light_handler(n))
        self.button4.clicked.connect(lambda checked, n=4: self.light_handler(n))
        self.button5.clicked.connect(lambda checked, n=5: self.light_handler(n))
        self.button6.clicked.connect(lambda checked, n=6: self.light_handler(n))
        self.button7.clicked.connect(lambda checked, n=7: self.light_handler(n))
        self.button8.clicked.connect(lambda checked, n=8: self.light_handler(n))

        self.allOFFButton.clicked.connect(lambda checked, n=13: self.light_handler(n))
        self.allONButton.clicked.connect(lambda checked, n=14: self.light_handler(n))

        self.leftButton.clicked.connect(lambda checked, n=9: self.button_clicked_handler(n))
        self.rightButton.clicked.connect(lambda checked, n=10: self.button_clicked_handler(n))
        self.upButton.clicked.connect(lambda checked, n=11: self.button_clicked_handler(n))
        self.downButton.clicked.connect(lambda checked, n=12: self.button_clicked_handler(n))

        self.stopButton.clicked.connect(lambda checked, n=15: self.button_clicked_handler(n))

        self.stepRight.clicked.connect(lambda checked, n=16: self.button_clicked_handler(n))
        self.stepLeft.clicked.connect(lambda checked, n=17: self.button_clicked_handler(n))
        self.stepUp.clicked.connect(lambda checked, n=18: self.button_clicked_handler(n))
        self.stepDown.clicked.connect(lambda checked, n=19: self.button_clicked_handler(n))

        self.pushButton.clicked.connect(lambda checked, n=20: self.button_clicked_handler(n))

    def light_handler(self, id):
        self.button_clicked_handler(id)
        self.label.setText(translate_light_status[ledStatus[0]])
        self.label_4.setText(translate_light_status[ledStatus[1]])
        self.label_6.setText(translate_light_status[ledStatus[2]])
        self.label_8.setText(translate_light_status[ledStatus[3]])
        self.label_9.setText(translate_light_status[ledStatus[4]])
        self.label_12.setText(translate_light_status[ledStatus[5]])
        self.label_14.setText(translate_light_status[ledStatus[6]])
        self.label_16.setText(translate_light_status[ledStatus[7]])
    
    def button_clicked_handler(self,id):
        global s
        if id <= 12:
            ret = s.send((translation[id]).encode())
            if id <= 8:
                ledStatus[id-1] = not ledStatus[id-1] 
                print(ledStatus)
        elif id == 13:
            for i in range(8):
                if ledStatus[i]:
                    ret = s.send((translation[i+1]).encode())
                    ledStatus[i] = False
                    time.sleep(0.3)
                    print(ledStatus)

        elif id == 14: 
            for i in range(4):
                if not ledStatus[i]:
                    ret = s.send((translation[i+1]).encode())
                    time.sleep(0.3)
                    ledStatus[i] = True
                    print(ledStatus)

                if ledStatus[7-i]: 
                    ret = s.send((translation[8-i]).encode())
                    time.sleep(0.3)
                    ledStatus[7-i] = False
                    print(ledStatus)
        
        elif id == 15:
            ret = s.send(b'0')
            print(ledStatus)

        elif id >=16 and id <= 20:
            # Step command or check vibr
            ret = s.send((translation[id]).encode())
            val = s.recv(1)
            val = bool(int.from_bytes(val, "big"))
            if val:
                self.label_19.setText("NO")
            else:
                self.label_19.setText("YES")

            print(val)

        print(f"Button name clicked: {translation[id]}")


class StepWindow(QMainWindow, Ui_StepWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.send_button.clicked.connect(self.get_step_data)

    def get_step_data(self):
        x_step = self.x_step_value.value()
        y_step = self.y_step_value.value()
        delta = self.y_step_value_3.value()
        threshold = self.y_step_value_2.value()
        print(self.x_step_value.value())
        print(self.y_step_value.value())

        ######## communicate the values formatted correctly
        #s.send(f"x{x_step}y{y_step}d{delta}t{threshold}\n".encode())
        
        self.close()
        self.mainwind = Window()
        self.mainwind.show()
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    global s
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("192.48.56.2",80))
    time.sleep(2) 
    print(s)
    
    win = StepWindow()
    win.show()
    sys.exit(app.exec())


    s.close()