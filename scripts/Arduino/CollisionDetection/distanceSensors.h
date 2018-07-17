#ifndef DISTANCESENSORS_H
#define DISTANCESENSORS_H

typedef unsigned char byte;   // weird??

#include <SoftwareSerial.h>
#include <Wire.h>

// Ultra sonic SRF01 ------------------------------------------------------------------
//#include <SoftwareSerial.h>

#define sonarSerialPin   53
#define ultraSonicAdress_01        0x01
#define ultraSonicAdress_02        0x02
#define ultraSonicAdress_03        0x03
#define ultraSonicAdress_04        0x04

//SoftwareSerial sonars = SoftwareSerial(sonarSerialPin, sonarSerialPin);

// distances
//int uSDist01;



// Explanation for all the externs:
//https://arduino.stackexchange.com/questions/39502/multiple-definition-of-compiler-error

extern int uSDist01;
extern int uSDist02;
extern int uSDist03;
extern int uSDist04;

//int uSDist02;
//int uSDist03;
//int uSDist04;

extern int uSMin;


#define GETRANGE         0x54                                       // Byte used to get range from SRF01
#define GETSTATUS        0x5F                                       // Byte used to get the status of the transducer
#define GETSOFT          0x5D                                       // Byte to tell SRF01 we wish to read software version
//SoftwareSerial sonars = SoftwareSerial(sonarSerialPin, sonarSerialPin);

// Ultra sonic SRF01 ------------------------------------------------------------------

// Infrared ------------------------------------------------------------------
#define iRRightPin   8
#define iRMiddlePin   7
#define iRLeftPin   6


//const int frameSize = 1;

#define frameSize 1
extern float iRVoltageRight[frameSize];
extern float iRVoltageMiddle[frameSize];
extern float iRVoltageLeft[frameSize];

// distances
extern float iRDistRight;   // right
extern float iRDistMiddle;   // middle
extern float iRDistLeft;   // left

extern float iRDistRightNotLin;   // right
extern float iRDistMiddleNotLin;   // middle
extern float iRDistLeftNotLin;   // left

extern int iRMin;

void resetAllDistances();

// send a request to the ultrasonic sensors and update 
// the current distances
void getUltrasonicDistances();

// expects voltage in int actually
float iR_GP2Y0A21(float voltage);

float iR_GP2D120(float voltage);

float iR_GP2Y0A60S(float voltage);

// gets the current voltage on the specified infra Red Pin
float getIRVoltage(int pin);

void getInfraredDistances();

// Input: address of the sensort
// Output: distance to an obstacle
int getUltraSonicRange(int address);

void initSonarSensor(int address);

void SRF01_Cmd(byte Address, byte cmd);


void getInfraredNotLin();
#endif
