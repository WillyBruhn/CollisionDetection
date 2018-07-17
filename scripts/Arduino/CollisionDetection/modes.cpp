
#include <SoftwareSerial.h>
#include <Wire.h>

#include "modes.h"

#include "display.h"  // need this to use millis() here

#include "driveBehaviour.h"


enum mode currentMode = DRIVING;
bool RUN;
enum DrivingMode currentDrivingMode = STILL;
enum DrivingMode nextDrivingMode = STILL;

unsigned long startTimeOfCurrentMode = 0;
unsigned long endTimeOfCurrentMode = 0;

void nextMode(){
  currentMode = (mode) currentMode + 1;
  if(currentMode > modesNum) currentMode = 0;
  RUN = false;  // better save than sorry
}

/*
 * Returns the time in milis() that we stayed in the current mode.
 * Uses "startTimeOfCurrentMode"
 */
int timeInCurrentMode(){
  return(millis() - startTimeOfCurrentMode);  
}

/*
 * Switch to DrivingMode dr for t milliseconds
 */
void switchToDrivingMode(DrivingMode dr, int t, DrivingMode dr2){
  startTimeOfCurrentMode = millis();
  endTimeOfCurrentMode = startTimeOfCurrentMode + t;

  currentDrivingMode = dr;
  nextDrivingMode = dr2;
}

/*
 * Switch to DrivingMode dr for t milliseconds
 * and switch back to the current mode afterwards.
 */
void switchToDrivingMode(DrivingMode dr, int t){
  switchToDrivingMode(dr,t,currentDrivingMode);
}


/*
 * Switches to the next drivingMode if the time of the 
 * current driving mode has run out.
 */
void switchToNextDrivingMode(){
  if(millis() > endTimeOfCurrentMode && endTimeOfCurrentMode != 0){
    currentDrivingMode = nextDrivingMode;
    endTimeOfCurrentMode = 0;   // stay in this mode until anything interrupts
    startTimeOfCurrentMode = millis();
  }
}

/*
 * Depending on the currentDrivingMode the robot will move in a certain direction.
 * 
 */
void drive(){
    switch(currentDrivingMode){
    case STILL : 
                  rmspeed = 0;
                  lmspeed = 0;
      break;
    case FORWARD : 
                  rmspeed = maxSpeedRight;
                  lmspeed = maxSpeedLeft;
      break;
    case LEFT : 
                  rmspeed = maxSpeedRight;
                  lmspeed = -maxSpeedLeft;
      break;
    case RIGHT : 
                  rmspeed = -maxSpeedRight;
                  lmspeed = maxSpeedLeft;
      break;
    case BACKWARD : 
                  rmspeed = -maxSpeedRight;
                  lmspeed = -maxSpeedLeft;
      break;
  }
}


/*
 * For Debugging. Prints the currentdriving mode to serialport.
 */
void printDrivingMode(){
      switch(currentDrivingMode){
    case STILL : Serial.println("STILL");
      break;
    case FORWARD : Serial.println("FORWARD");
      break;
    case LEFT : Serial.println("LEFT");
      break;
    case RIGHT : Serial.println("RIGHT");
      break;
    case BACKWARD : Serial.println("BACKWARD");
      break;
  }
}



