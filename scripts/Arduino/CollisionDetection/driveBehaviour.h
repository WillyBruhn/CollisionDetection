#ifndef DRIVEBEHAVIOUR_H
#define DRIVEBEHAVIOUR_H

#include <SoftwareSerial.h>

#if defined(ARDUINO) && ARDUINO >= 100
  #include "Arduino.h"
  #else
  #include "WProgram.h"
  #endif
#include <Wire.h>


typedef unsigned char byte;   // weird??

// TREX ---------------------------------------------------------------------------------
#define startbyte 0x0F
#define I2Caddress 0x07

extern int sv[6];                 // servo positions: 0 = Not Used
extern int sd[6];                      // servo sweep speed/direction
extern int lmspeed,rmspeed;                                 // left and right motor speed from -255 to +255 (negative value = reverse)
extern int ldir;                                          // how much to change left  motor speed each loop (use for motor testing)
extern int rdir;                                          // how much to change right motor speed each loop (use for motor testing)
extern byte lmbrake,rmbrake;                                // left and right motor brake (non zero value = brake)
extern byte devibrate;                                   // time delay after impact to prevent false re-triggering due to chassis vibration
extern int sensitivity;                                  // threshold of acceleration / deceleration required to register as an impact
extern int lowbat;                                      // adjust to suit your battery: 550 = 5.50V
extern byte i2caddr;                                      // default I2C address of T'REX is 7. If this is changed, the T'REX will automatically store new address in EEPROM
extern byte i2cfreq;                                      // I2C clock frequency. Default is 0=100kHz. Set to 1 for 400kHz


extern int maxSpeedLeft;
extern int maxSpeedRight;

extern int maxSpeed;

extern bool colided;


typedef unsigned long myInt;
extern myInt colision_time;

 extern myInt time_to_stand_still;
extern myInt time_to_drive_back ;
 extern myInt time_since_colision;

extern myInt current_time;

extern bool forward;


typedef double myFloat;

//------------------------------------------------------

bool enoughDistance();

bool enoughDistanceInFront();

void turnLeft();

void processDistances();


// take dist from Left-Right
// and add this dist
void corridorBehaviour();


void corridorBehaviour2(bool collectData, const unsigned long &avg_loop_lenght);

void corridorBehaviour3(bool collectData);

void driveForwardTillCollision();

void driveRectangle();

void setupBehaviour();

void driveForwardAndBackward();

void driveForward();

void driveForwardAndBreak();

void forwardAndBackward(bool collectData, int val);

void forwardAfterCollision();

//void getCollisionDirection(myFLoat x, myFloat y, myFloat z, myFLoat &x_out, myFloat &y_out, myFloat &z_out);

// void stopTREX();

myFloat normOfVector(myFloat x, myFloat y, myFloat z);

#endif
