#include "distanceSensors.h"
#include <math.h>

#include "utility.h"

extern SoftwareSerial sonars;

float iRDistRight;   // right
float iRDistMiddle;   // middle
float iRDistLeft;   // left

int iRMin = 25;

float iRVoltageRight[frameSize];
float iRVoltageMiddle[frameSize];
float iRVoltageLeft[frameSize];

float iRDistRightNotLin;   // right
float iRDistMiddleNotLin;   // middle
float iRDistLeftNotLin;   // left

//#include <Wire.h>

//#include <wiring.h"


#if defined(ARDUINO) && ARDUINO >= 100
  #include "Arduino.h"
  #else
  #include "WProgram.h"
  #endif
#include <Wire.h>

void resetAllDistances(){
    uSDist01 = 0;
    uSDist02 = 0;
    uSDist03 = 0;
    uSDist04 = 0;

    iRDistRight = 0;
    iRDistMiddle = 0;
    iRDistLeft = 0;
}

// send a request to the ultrasonic sensors and update 
// the current distances
void getUltrasonicDistances(){
  uSDist01 = getUltraSonicRange(ultraSonicAdress_01);
  uSDist02 = getUltraSonicRange(ultraSonicAdress_02);
  uSDist03 = getUltraSonicRange(ultraSonicAdress_03);
  uSDist04 = getUltraSonicRange(ultraSonicAdress_04);
}

// expects voltage in int actually
float iR_GP2Y0A21(float voltage){
  // last formula from
  // Linearizing-Sharp-IR-Sensor-Data.pdf
  // /home/willy/10.Semester/Robotik/PatricksOrdner14.5/robotik/Hardware/IrSensor
  // return 2914/(voltage + 5.0) - 1;

  float b = 3.5;
  float k = 0.42;
  float m = -13;

  // return 1/(m*voltage + b) - k;


  return 6787/(voltage -3.0) - 4.0;
}

float iR_GP2D120(float voltage){
  // last formula from
  // Linearizing-Sharp-IR-Sensor-Data.pdf
  // /home/willy/10.Semester/Robotik/PatricksOrdner14.5/robotik/Hardware/IrSensor
  return 2914/(voltage + 5.0) - 1;
}

float iR_GP2Y0A60S(float voltage){

  // http://www.instructables.com/id/How-to-setup-a-Pololu-Carrier-with-Sharp-GP2Y0A60S/
  return 187754 * pow(voltage, -1.51);
}



// gets the current voltage on the specified infra Red Pin
float getIRVoltage(int pin){

  // return analogRead(pin) * (5.0 / 1023.0);

  return analogRead(pin);
}


void getInfraredDistances(){

  appendToArray(iRVoltageRight, frameSize, getIRVoltage(iRRightPin));
  iRDistRight = iR_GP2Y0A21(getAverage(iRVoltageRight, frameSize));

  appendToArray(iRVoltageMiddle, frameSize, getIRVoltage(iRMiddlePin));
  iRDistMiddle = iR_GP2Y0A60S(getAverage(iRVoltageMiddle, frameSize));
  //iRDistMiddle = iR_GP2Y0A21(getAverage(iRVoltageMiddle, frameSize));

  //Serial.println(iRDistMiddle);

  appendToArray(iRVoltageLeft, frameSize, getIRVoltage(iRLeftPin));
  iRDistLeft = iR_GP2Y0A21(getAverage(iRVoltageLeft, frameSize));


  // printArray2Console(iRVoltageRight, frameSize);

/*
  iRDistRight = iR_GP2Y0A21(getIRVoltage(iRRightPin));
  iRDistMiddle = iR_GP2D120(getIRVoltage(iRMiddlePin));
  iRDistLeft = iR_GP2Y0A21(getIRVoltage(iRLeftPin));

  */
}

// Input: address of the sensort
// Output: distance to an obstacle
int getUltraSonicRange(int address){

    // double infraMiddleReading= analogRead(7)/1024.0;
    //Serial.println(infraMiddleReading);
    //delay(100);
  
    byte hByte, lByte, statusByte, b1, b2, b3;  
    SRF01_Cmd(address, GETRANGE);                       // Get the SRF01 to perform a ranging and send the data back to the arduino  
    while (sonars.available() < 2);
    hByte = sonars.read();                                   // Get high byte
    lByte = sonars.read();                                   // Get low byte
    int range = ((hByte<<8)+lByte);                         // Put them together

    //Serial.print("Adress ");
    //Serial.print(address);
    //Serial.print(" ");
    //Serial.println(range);
                                
    //Serial.println(range, DEC);                                // Print range result to the screen
    //Serial.println("  ");                                      // Print some spaces to the screen to make sure space direcly after the result is clear
    
    return range;
}


void initSonarSensor(int address){
  
  byte softVer;
  SRF01_Cmd(address, GETSOFT);                        // Request the SRF01 software version
  //while (sonartest.available() < 1);
    softVer = sonars.read(); 
}


void SRF01_Cmd(byte Address, byte cmd){               // Function to send commands to the SRF01
  pinMode(sonarSerialPin, OUTPUT);
  digitalWrite(sonarSerialPin, LOW);                        // Send a 2ms break to begin communications with the SRF01                         
  delay(2);                                               
  digitalWrite(sonarSerialPin, HIGH);                            
  delay(1);                                                
  sonars.write(Address);                               // Send the address of the SRF01
  sonars.write(cmd);                                   // Send commnd byte to SRF01
  pinMode(sonarSerialPin, INPUT);
  int availbleJunk = sonars.available();               // As RX and TX are the same pin it will have recieved the data we just sent out, as we dont want this we read it back and ignore it as junk before waiting for useful data to arrive
  for(int x = 0;  x < availbleJunk; x++) byte junk = sonars.read();
}


void getInfraredNotLin(){
   iRDistLeftNotLin = analogRead(iRLeftPin);
   iRDistMiddleNotLin = analogRead(iRMiddlePin);
   iRDistRightNotLin = analogRead(iRRightPin);
}
