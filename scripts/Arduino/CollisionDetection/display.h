#ifndef DISPLAY_H
#define DISPLAY_H

#include "modes.h"
#include "Adafruit_LiquidCrystal.h"

#include "distanceSensors.h"

// Connect via i2c, default address #0 (A0-A2 not jumpered)
extern Adafruit_LiquidCrystal lcd;


void lcdDisplayMode(int line);

void serialDisplaySpeed();

// function to display the ultra sonic sensor measurements on the
// lcd backpack
void displayUltrasonicSensors();

// display the uS in given line
void displayUltrasonicSensors(int line);
// display the iR in given line
void displayInfraredDistances(int line);

void displayMotor(int line);

void displayStatus();

void displayInfraredDistancesNotLin(int line);

#endif
