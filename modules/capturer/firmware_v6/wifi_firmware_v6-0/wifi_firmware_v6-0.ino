#include "WiFiS3.h"
#include "arduino_secrets.h"
#include <AccelStepper.h>

#define STOPPER_ACT HIGH
#include <Wire.h>

// MPU-9250 I2C address (depends on AD0 pin, here we assume it's connected to GND)
const int MPU9250_ADDRESS = 0x68;
/////////////////////////////////////////////////////////////////////
int x_motor_pin = 13;
int x_dir_pin = 12;
int y_motor_pin = 11;
int y_dir_pin = 10;

AccelStepper x_stepper(AccelStepper::DRIVER, x_motor_pin, x_dir_pin);
AccelStepper y_stepper(AccelStepper::DRIVER, y_motor_pin, y_dir_pin);

int x_stopper = 9;
int y_stopper = 8;


int ledPins[] = {
  7, 6, 5, 4, 3, 2, 1, 0
};
bool ledStatus[] = {
  false, false, false, false, false, false, false, false
};
int pinCount = 8;
char inChar = '0';
int x_velocity = 500;
int y_velocity = 500;

bool moving = false;


char ssid[] = SECRET_SSID;  // your network SSID (name)
char pass[] = SECRET_PASS;  // your network password (use for WPA, or use as key for WEP)



int x_step = 149;
int y_step = 894;

float x_cm;
float y_cm;
float delta;
float threshold;

int keyIndex = 0;  // your network key index number (needed only for WEP)
bool ignore_x = false;
bool ignore_y = false;
char command;

int status = WL_IDLE_STATUS;
WiFiServer server(80);



long XMAX = 17900;
long YMAX = -96900;



void setupMPU9250() {
  // Wake up MPU-9250
  writeMPU9250(MPU9250_ADDRESS, 0x6B, 0x00);
  // Configure gyroscope and accelerometer
  // This is a basic configuration and should be adjusted for your application
  writeMPU9250(MPU9250_ADDRESS, 0x1B, 0x00); // Set gyroscope to ±250 degrees/sec
  writeMPU9250(MPU9250_ADDRESS, 0x1C, 0x00); // Set accelerometer to ±2g
}


void X_calibr() {
  x_stepper.setSpeed(-x_velocity);
  while (digitalRead(x_stopper) != STOPPER_ACT) {
    x_stepper.runSpeed();
  }
}

void Y_calibr() {
  y_stepper.setSpeed(y_velocity);
  while (digitalRead(y_stopper) != STOPPER_ACT) {
    y_stepper.runSpeed();
  }
}

void setup() {
  //Initialize serial and wait for port to open:
  Serial.begin(9600);
  Wire.begin();
  setupMPU9250();

  while (!Serial) {
    ;  // wait for serial port to connect. Needed for native USB port only
  }
  Serial.println("Access Point Web Server");

  //// setup HW
  pinMode(x_motor_pin, OUTPUT);
  pinMode(x_dir_pin, OUTPUT);
  pinMode(x_stopper, INPUT);

  pinMode(y_dir_pin, OUTPUT);
  pinMode(y_motor_pin, OUTPUT);
  pinMode(y_stopper, INPUT);



  for (int thisPin = 0; thisPin < pinCount; thisPin++) {
    pinMode(ledPins[thisPin], OUTPUT);
  }


  // check for the WiFi module:
  if (WiFi.status() == WL_NO_MODULE) {
    Serial.println("Communication with WiFi module failed!");
    // don't continue
    while (true)
      ;
  }

  String fv = WiFi.firmwareVersion();
  if (fv < WIFI_FIRMWARE_LATEST_VERSION) {
    Serial.println("Please upgrade the firmware");
  }

  // by default the local IP address will be 192.168.4.1
  // you can override it with the following:
  WiFi.config(IPAddress(192, 48, 56, 2));

  // print the network name (SSID);
  Serial.print("Creating access point named: ");
  Serial.println(ssid);

  // Create open network. Change this line if you want to create an WEP network:
  status = WiFi.beginAP(ssid, pass);
  if (status != WL_AP_LISTENING) {
    Serial.println("Creating access point failed");
    // don't continue
    while (true)
      ;
  }

  // wait 10 seconds for connection:
  delay(10000);

  // start the web server on port 80
  server.begin();

  // you're connected now, so print out the status
  printWiFiStatus();


  x_stepper.setMaxSpeed(1000);
  y_stepper.setMaxSpeed(1000);

  for (int thisPin = 0; thisPin < pinCount; thisPin++) {
    digitalWrite(ledPins[thisPin], false);
    ledStatus[thisPin] = false;
  }


  command = '0';
  Serial.println("Setup ended.");
}


void loop() {



  // compare the previous status to the current status

  if (status != WiFi.status()) {
    // it has changed update the variable
    status = WiFi.status();

    if (status == WL_AP_CONNECTED) {
      // a device has connected to the AP
      Serial.println("Device connected to AP");
    } else {
      // a device has disconnected from the AP, and we are back in listening mode
      Serial.println("Device disconnected from AP");
    }
  }


  WiFiClient client = server.available();

  // If you get a client, you need to check the position and set all the lights on to have a known initial state.
  // this is not possible using serial communication, since there is no possibility to check new clients automatically
  // and since I have no need up to now, i will simply comment this part and assuming it done in the setup()

  if (client) {

    //new cient
    Serial.println("new client");
    for (int thisPin = 0; thisPin < pinCount; thisPin++) {
      digitalWrite(ledPins[thisPin], false);
      ledStatus[thisPin] = false;
    }
    Serial.println("All LED up");
    X_calibr();
    x_stepper.setCurrentPosition(0);
    Y_calibr();
    y_stepper.setCurrentPosition(0);

    while (client.connected()) {


      // Check if there are data to read
      //if any, set the command

      // This is required for the Arduino Nano RP2040 Connect - otherwise it will loop so fast that SPI will never be served.
      if (client.available()) {

        // Read the byte in input
        char inChar = (char)client.read();


        // Check the input

        if (inChar == 'w') {
          if (command = '0') {
            Serial.println("Speed set");
            y_stepper.setSpeed(-y_velocity*3);
            x_stepper.setSpeed(0);
          }

          ignore_y = true;
          command = 'w';
        } else if (inChar == 's') {
          if (command = '0') {
            Serial.println("Speed set");
            y_stepper.setSpeed(y_velocity*3);
            x_stepper.setSpeed(0);
          }
          moving = true;

          ignore_y = false;
          command = 's';
        } else if (inChar == 'd') {
          if (command = '0') {
            Serial.println("Speed set");
            y_stepper.setSpeed(0);
            x_stepper.setSpeed(x_velocity*3);
          }

          ignore_x = true;
          command = 'd';
        } else if (inChar == 'a') {
          if (command = '0') {
            Serial.println("Speed set");
            y_stepper.setSpeed(0);
            x_stepper.setSpeed(-x_velocity*3);
          }

          ignore_x = false;
          command = 'a';

        } else if (inChar <= '8') {
          char i = inChar - '1';
          ledStatus[i] = !ledStatus[i];
          digitalWrite(ledPins[i], ledStatus[i]);

          //Serial.print("Change status");
          ////Serial.print(ledPins[i]);
          //Serial.print(" goes to ");
          //Serial.println(ledStatus[i]);
          x_stepper.setSpeed(0);
          y_stepper.setSpeed(0);
          command = '0';

        }

        else if (inChar == 'i') {
          if (y_stepper.currentPosition() - y_cm * y_step > YMAX && y_stepper.currentPosition() - y_cm * y_step <= 0) {
            
            y_stepper.setSpeed(y_velocity);
            y_stepper.setAcceleration(50);
            y_stepper.move(-y_cm * y_step);
            y_stepper.runToPosition();
            Serial.println(y_stepper.currentPosition());
            Serial.println("Check vibrs for stepUP");
            client.write((int)checkVibrations(delta, threshold));
            Serial.println("Check concluded");
          }

          command = '0';
        } else if (inChar == 'k') {
          if (y_stepper.currentPosition() + y_cm * y_step > YMAX && y_stepper.currentPosition() + y_cm * y_step <= 0) {
            
            y_stepper.setSpeed(-y_velocity);
            y_stepper.setAcceleration(50);
            y_stepper.move(y_cm * y_step);
            y_stepper.runToPosition();
            Serial.println(y_stepper.currentPosition());
            Serial.println("Check vibrs for stepDown");
            client.write((int)checkVibrations(delta, threshold));
            Serial.println("Check concluded");
          }


          command = '0';
        } else if (inChar == 'l') {
          if (x_stepper.currentPosition() + x_cm * x_step < XMAX && x_stepper.currentPosition() + x_cm * x_step >= 0) {
            x_stepper.setSpeed(x_velocity);
            x_stepper.setAcceleration(100);
            x_stepper.move(x_cm * x_step);
            x_stepper.runToPosition();
            Serial.println(x_stepper.currentPosition());
            Serial.println("Check vibrs for stepRight");
            client.write((int)checkVibrations(delta, threshold));
            Serial.println("Check concluded");
          }
        

          command = '0';
        } else if (inChar == 'j') {
          if (x_stepper.currentPosition() - x_cm * x_step < XMAX && x_stepper.currentPosition() - x_cm * x_step >= 0) {
            x_stepper.setSpeed(-x_velocity);
            x_stepper.setAcceleration(100);
            x_stepper.move(-x_cm * x_step);
            x_stepper.runToPosition();
            Serial.println(x_stepper.currentPosition());

            Serial.println("Check vibrs for stepLeft");
            client.write((int)checkVibrations(delta, threshold));
            Serial.println("Check concluded");
          }


          command = '0';

        } else if (inChar == 'x') {
          Serial.println("Start checking parameters");
          String inputString = "";
          char rChar = '0';
          while(rChar != '\n'){
            rChar = (char)client.read();
            inputString += rChar;
            
          }
          Serial.println(inputString);
          char char_y = inputString.indexOf('y');
          char char_d = inputString.indexOf('d');
          char char_thr = inputString.indexOf('t');
          Serial.println("2");
          x_cm = inputString.substring(0, char_y).toFloat();
          Serial.println("3");
          y_cm = inputString.substring(char_y + 1, char_d).toFloat();
          delta = inputString.substring(char_d + 1, char_thr).toFloat();
          threshold = inputString.substring(char_thr + 1).toFloat();
          Serial.print("X=");Serial.print(x_cm);
          Serial.print(", Y=");Serial.print(y_cm);
          Serial.print(", DELTA=");Serial.print(delta);
          Serial.print(", Threshold=");Serial.print(threshold);
          Serial.println();

          //FOR DEBUG PURPOSES, REMOVE IT AFTER TESTING
          //checkVibrations(delta, threshold);
          
          command = '0';

        } else if (inChar == 'c'){
          // check vibration
          Serial.println("Check vibrs on demand");
          client.write((int)checkVibrations(delta, threshold));
          Serial.println("Check concluded");
        } 
        else {
          command = '0';
          moving = false;
          x_stepper.setSpeed(0);
          y_stepper.setSpeed(0);
        }
      }



      //check stoppers
      // todo check max pos
      if ((digitalRead(y_stopper) == STOPPER_ACT && command == 's') || (y_stepper.currentPosition() <= YMAX && command == 'w')) {
        y_stepper.setSpeed(0);
      }
      if ((digitalRead(x_stopper) == STOPPER_ACT && command == 'a') || (x_stepper.currentPosition() >= XMAX && command == 'd')) {
        x_stepper.setSpeed(0);
      }
      //run speed
      y_stepper.runSpeed();
      x_stepper.runSpeed();
    }
  }
}

void printWiFiStatus() {
  // print the SSID of the network you're attached to:
  Serial.print("SSID: ");
  Serial.println(WiFi.SSID());

  // print your WiFi shield's IP address:
  IPAddress ip = WiFi.localIP();
  Serial.print("IP Address: ");
  Serial.println(ip);

  // print where to go in a browser:
  Serial.print("To see this page in action, open a browser to http://");
  Serial.println(ip);
}


void response(WiFiClient client) {

  client.println("HTTP/1.1 200 OK \
  Content-type:text/html \
    \
  <p style=\"font-size:7vw;\">Click <a href=\"/H\">here</a> turn the LED on<br></p> \
  <p style=\"font-size:7vw;\">Click <a href=\"/L\">here</a> turn the LED off<br></p> \
  ");
}

void readSensorData() {
  const int ACCEL_XOUT_H = 0x3B; // Register address for accelerometer X-axis high byte

  Wire.beginTransmission(MPU9250_ADDRESS);
  Wire.write(ACCEL_XOUT_H); // Start reading at ACCEL_XOUT_H
  Wire.endTransmission(false); // End transmission but keep I2C active

  Wire.requestFrom(MPU9250_ADDRESS, 14); // Request 14 bytes (6 for accelerometer, 2 for temperature, 6 for gyroscope)
  if (Wire.available() == 14) {
    int16_t accelX = (Wire.read() << 8) | Wire.read();
    int16_t accelY = (Wire.read() << 8) | Wire.read();
    int16_t accelZ = (Wire.read() << 8) | Wire.read();
    int16_t temp = (Wire.read() << 8) | Wire.read(); // Temperature data
    int16_t gyroX = (Wire.read() << 8) | Wire.read();
    int16_t gyroY = (Wire.read() << 8) | Wire.read();
    int16_t gyroZ = (Wire.read() << 8) | Wire.read();

    // Convert raw data to meaningful units if needed
    float accelX_g = accelX / 16384.0; // Assuming ±2g range
    float accelY_g = accelY / 16384.0;
    float accelZ_g = accelZ / 16384.0;
    float gyroX_dps = gyroX / 131.0; // Assuming ±250 degrees/sec
    float gyroY_dps = gyroY / 131.0;
    float gyroZ_dps = gyroZ / 131.0;

    float acc_mod, gyro_mod;
    acc_mod = sqrt(accelX_g*accelX_g + accelY_g*accelY_g + accelZ_g*accelZ_g);
    gyro_mod = sqrt(gyroX_dps*gyroX_dps + gyroY_dps*gyroY_dps + gyroZ_dps*gyroZ_dps);
    // Print the values to the Serial Monitor
    Serial.print("Accel (g): mod=");
    Serial.print(acc_mod);
    Serial.print(", Gyro : mod=");
    Serial.println(gyro_mod);

  } 
  else {
    Serial.println("Error reading sensor data!");
  }
}

void writeMPU9250(byte address, byte reg, byte data) {
  Wire.beginTransmission(address);
  Wire.write(reg);
  Wire.write(data);
  Wire.endTransmission();
}

float getGyroModule() {
  const int ACCEL_XOUT_H = 0x3B; // Register address for accelerometer X-axis high byte

  Wire.beginTransmission(MPU9250_ADDRESS);
  Wire.write(ACCEL_XOUT_H); // Start reading at ACCEL_XOUT_H
  Wire.endTransmission(false); // End transmission but keep I2C active

  Wire.requestFrom(MPU9250_ADDRESS, 14); // Request 14 bytes (6 for accelerometer, 2 for temperature, 6 for gyroscope)
  if (Wire.available() == 14) {
    int16_t accelX = (Wire.read() << 8) | Wire.read();
    int16_t accelY = (Wire.read() << 8) | Wire.read();
    int16_t accelZ = (Wire.read() << 8) | Wire.read();
    int16_t temp = (Wire.read() << 8) | Wire.read(); // Temperature data
    int16_t gyroX = (Wire.read() << 8) | Wire.read();
    int16_t gyroY = (Wire.read() << 8) | Wire.read();
    int16_t gyroZ = (Wire.read() << 8) | Wire.read();

    // Convert raw data to meaningful units if needed
    float accelX_g = accelX / 16384.0; // Assuming ±2g range
    float accelY_g = accelY / 16384.0;
    float accelZ_g = accelZ / 16384.0;
    float gyroX_dps = gyroX / 131.0; // Assuming ±250 degrees/sec
    float gyroY_dps = gyroY / 131.0;
    float gyroZ_dps = gyroZ / 131.0;

    float acc_mod, gyro_mod;
    //acc_mod = sqrt(accelX_g*accelX_g + accelY_g*accelY_g + accelZ_g*accelZ_g);
    gyro_mod = sqrt(gyroX_dps*gyroX_dps + gyroY_dps*gyroY_dps + gyroZ_dps*gyroZ_dps);
    // Print the values to the Serial Monitor
    return gyro_mod;

  } 
  else {
    Serial.println("Error reading sensor data!");
    return -1.0;
  }
}



bool checkVibrations(float delta_s, float threshold){
  /*
  1. wait 2 sec
  2. sampling every 0.2sec for delta time
  3. check if mean if maximum is under threshold*/
  
  float* sampled_values = nullptr;
  
  delay(2000);
  float sampling_interval = 200.0;
  float interval = delta_s*1000;
  int number_of_sampling = (int)(interval / sampling_interval); 
  Serial.print("# sampling=");Serial.println(number_of_sampling);
  if (number_of_sampling == 0){
    Serial.println("Time of sampling too low! Assuming no vibrations.");
    return true;
  }
  sampled_values = new float[number_of_sampling];
  if (sampled_values == nullptr){
    Serial.println("Memory allocation failed!");
    return false;
  }
  for (int i = 0; i < number_of_sampling; i++) {
    sampled_values[i] = getGyroModule();
    delay(200);
  }

  float tot = 0;
  for (int i = 0; i < number_of_sampling; i++) {
    tot = tot +sampled_values[i];
  }
  float mean = tot / number_of_sampling;
  Serial.print("MEAN: ");Serial.print(mean);
  Serial.print(" thr: ");Serial.println(threshold);
  if (mean > threshold){
    Serial.println("Return FALSE!");
    return false;
  }
  else{    
    Serial.println("Return TRUE!");
    return true;
  }

}