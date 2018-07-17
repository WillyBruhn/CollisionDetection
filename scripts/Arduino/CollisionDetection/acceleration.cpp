#ifndef ACC_SEN_H
#define ACC_SEN_H

//#include "defs.h"
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>  // acceleration sensor

#include "display.h"

Adafruit_BNO055 bno = Adafruit_BNO055(55); 

const int OPERATION_MODE_NDOF = 0X0C; //schneller modus mit accel/eulor und gyro

double accX = 0.0f;
double accY = 0.0f;
double accZ = 0.0f;

double gyroX = 0.0f;
double gyroY = 0.0f;
double gyroZ = 0.0f;


#define INIT_REFRESH_RATE 100

/**
* Sets up the acceleration sensor and displays basic information about the acceleration sensor on the given lcd display.
*/
void initAccSen()
{
  lcd.clear();
  lcd.print("Setup acc-sen:");
  lcd.setCursor(0,1);
  if(!bno.begin())
  {
    /* There was a problem detecting the BNO055 ... check your connections */
    lcd.print("Error: No BNO055 detected!");
    while(1);
  }

  sensor_t sensor;
  bno.getSensor(&sensor);
  //bno.setMode(OPERATION_MODE_NDOF);

  lcd.print  ("Name:   ");
  lcd.setCursor(11,1);
  lcd.print(sensor.name);

  lcd.setCursor(0,2);
  lcd.print  ("Version:  ");
  lcd.setCursor(11,2);
  lcd.print(sensor.version);

  lcd.setCursor(0,3);
  lcd.print  ("Unique ID:");
  lcd.setCursor(11,3);
  lcd.print(sensor.sensor_id);

  delay(INIT_REFRESH_RATE);

  lcd.clear();

  lcd.setCursor(0,0);
  lcd.print  ("Max value:");
  lcd.setCursor(11,0);
  lcd.print(sensor.max_value);

  lcd.setCursor(0,1);
  lcd.print  ("Min value:");
  lcd.setCursor(11,1);
  lcd.print(sensor.min_value);

  lcd.setCursor(0,2);
  lcd.print  ("Resul.:  ");
  lcd.setCursor(11,2);
  lcd.print(sensor.resolution);

  delay(INIT_REFRESH_RATE);
}

/*
 * Reads out the current sensor status
 */
imu::Vector<3> lineacc;
void readAccSenStatus()
{
  lineacc = bno.getVector(Adafruit_BNO055::VECTOR_LINEARACCEL);
  accX = lineacc.x();
  accY = lineacc.y();
  accZ = lineacc.z();

  lineacc = bno.getVector(Adafruit_BNO055::VECTOR_GYROSCOPE);
  gyroX = lineacc.x();
  gyroY = lineacc.y();
  gyroZ = lineacc.z();
}


/*
 * Prints the current measured acceleration values to the lcd dispay (in the given line)
 *
 */
void printAccSenStatus(int line)
{
  lcd.setCursor(0,line);
  //lcd.print("acc: ");
  //lcd.setCursor(5,line);
  lcd.print(accX);
  lcd.setCursor(6,line);
  lcd.print(accY);
  lcd.setCursor(14,line);
  lcd.print(accZ);
}

/**************************************************************************/
/*
    Displays some basic information on this sensor from the unified
    sensor API sensor_t type (see Adafruit_Sensor for more information)
*/
/**************************************************************************/
void displaySensorDetails(Adafruit_BNO055& bno)
{
  sensor_t sensor;
  bno.getSensor(&sensor);
  Serial.println("------------------------------------");
  Serial.print  ("Sensor:       "); Serial.println(sensor.name);
  Serial.print  ("Driver Ver:   "); Serial.println(sensor.version);
  Serial.print  ("Unique ID:    "); Serial.println(sensor.sensor_id);
  Serial.print  ("Max Value:    "); Serial.print(sensor.max_value); Serial.println(" xxx");
  Serial.print  ("Min Value:    "); Serial.print(sensor.min_value); Serial.println(" xxx");
  Serial.print  ("Resolution:   "); Serial.print(sensor.resolution); Serial.println(" xxx");
  Serial.println("------------------------------------");
  Serial.println("");
  delay(500);
}

/**************************************************************************/
/*
    Display some basic info about the sensor status
*/
/**************************************************************************/
void displaySensorStatus(Adafruit_BNO055& bno)
{
  /* Get the system status values (mostly for debugging purposes) */
  uint8_t system_status, self_test_results, system_error;
  system_status = self_test_results = system_error = 0;
  bno.getSystemStatus(&system_status, &self_test_results, &system_error);

  /* Display the results in the Serial Monitor */
  Serial.println("");
  Serial.print("System Status: 0x");
  Serial.println(system_status, HEX);
  Serial.print("Self Test:     0x");
  Serial.println(self_test_results, HEX);
  Serial.print("System Error:  0x");
  Serial.println(system_error, HEX);
  Serial.println("");
  delay(500);
}

/**************************************************************************/
/*
    Display sensor calibration status
*/
/**************************************************************************/
void displayCalStatus(Adafruit_BNO055& bno)
{
  /* Get the four calibration values (0..3) */
  /* Any sensor data reporting 0 should be ignored, */
  /* 3 means 'fully calibrated" */
  uint8_t system, gyro, accel, mag;
  system = gyro = accel = mag = 0;
  bno.getCalibration(&system, &gyro, &accel, &mag);

  /* The data should be ignored until the system calibration is > 0 */
  Serial.print("\t");
  if (!system)
  {
    Serial.print("! ");
  }

  /* Display the individual values */
  Serial.print("Sys:");
  Serial.print(system, DEC);
  Serial.print(" G:");
  Serial.print(gyro, DEC);
  Serial.print(" A:");
  Serial.print(accel, DEC);
  Serial.print(" M:");
  Serial.print(mag, DEC);
}

#endif

