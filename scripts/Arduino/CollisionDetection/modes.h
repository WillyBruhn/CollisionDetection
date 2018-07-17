#include <stdbool.h>  // for bool

#ifndef MODES_H
#define MODES_H

const int modesNum = 3;
enum mode {NOTHING, DRIVING, MAXSPEED};

extern enum mode currentMode;

extern unsigned long startTimeOfCurrentMode;
extern unsigned long endTimeOfCurrentMode;


void nextMode();

// Button mode ---------------------------------------------------------------

enum DrivingMode {STILL, FORWARD, LEFT, RIGHT, BACKWARD};
extern enum DrivingMode currentDrivingMode;
extern enum DrivingMode nextDrivingMode;

extern bool RUN;          // start or stop the robot


int timeInCurrentMode();

//byte potentiometerPin = A11;

void switchToDrivingMode(DrivingMode dr, int t, DrivingMode dr2);
void switchToDrivingMode(DrivingMode dr, int t);

void switchToNextDrivingMode();

void drive();


void printDrivingMode();

#endif
