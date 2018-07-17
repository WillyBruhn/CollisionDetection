#ifndef ACC_SEN_H
#define ACC_SEN_H

//#include "defs.h"
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>  // acceleration sensor


extern Adafruit_BNO055 bno; 

extern float accX;
extern float accY;
extern float accZ;

extern float gyroX;
extern float gyroY;
extern float gyroZ;

/**
* Sets up the acceleration sensor and displays basic information about the acceleration sensor on the given lcd display.
*/
void initAccSen();

/*
 * Reads out the current sensor status
 */
void readAccSenStatus();

/*
 * Prints the current measured acceleration values to the lcd dispay (in the given line)
 *
 */
void printAccSenStatus(int line);


void sendAccSenStatus();

/**************************************************************************/
/*
    Displays some basic information on this sensor from the unified
    sensor API sensor_t type (see Adafruit_Sensor for more information)
*/
/**************************************************************************/
void displaySensorDetails(Adafruit_BNO055& bno);

/**************************************************************************/
/*
    Display some basic info about the sensor status
*/
/**************************************************************************/
void displaySensorStatus(Adafruit_BNO055& bno);

/**************************************************************************/
/*
    Display sensor calibration status
*/
/**************************************************************************/
void displayCalStatus(Adafruit_BNO055& bno);
#endif

