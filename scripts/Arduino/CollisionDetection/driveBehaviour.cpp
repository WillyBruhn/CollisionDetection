#include "driveBehaviour.h"
#include "distanceSensors.h"
#include "display.h"

#include "modes.h"

#include "acceleration.h"

#include "NN.h"

#include <SoftwareSerial.h>

#if defined(ARDUINO) && ARDUINO >= 100
  #include "Arduino.h"
  #else
  #include "WProgram.h"
  #endif
#include <Wire.h>



int sv[6]={1500,1500,1500,1500,0,0};                 // servo positions: 0 = Not Used
int sd[6]={5,10,-5,-15,20,-20};                      // servo sweep speed/direction
int lmspeed,rmspeed;                                 // left and right motor speed from -255 to +255 (negative value = reverse)
int ldir=5;                                          // how much to change left  motor speed each loop (use for motor testing)
int rdir=5;                                          // how much to change right motor speed each loop (use for motor testing)
byte lmbrake,rmbrake;                                // left and right motor brake (non zero value = brake)
byte devibrate=50;                                   // time delay after impact to prevent false re-triggering due to chassis vibration
int sensitivity=50;                                  // threshold of acceleration / deceleration required to register as an impact
int lowbat=550;                                      // adjust to suit your battery: 550 = 5.50V
byte i2caddr=7;                                      // default I2C address of T'REX is 7. If this is changed, the T'REX will automatically store new address in EEPROM
byte i2cfreq=0;                                      // I2C clock frequency. Default is 0=100kHz. Set to 1 for 400kHz


int maxSpeedLeft = 160;
int maxSpeedRight = 160;

int maxSpeed = 120;

bool colided = false;


void turnLeft(){
   rmspeed = 160;
   lmspeed = -170;
}

void processDistances(){
  if(enoughDistanceInFront()){
    RUN = true;

    rmspeed = maxSpeedRight;
    lmspeed = maxSpeedLeft;

  } else {

    turnLeft();
  }

}

myInt colision_time = 0;

myInt  time_to_stand_still = 400;
myInt  time_to_drive_back = 500;
myInt  time_to_turn = 500;

myInt time_since_colision = 0;

myInt current_time = 0;

bool forward = true;

void corridorBehaviour(){

  //current_time = millis();
  if(colided == true){
    //colision_time = millis();
    colided = false;
    forward = !forward;
  }

  time_since_colision = current_time - colision_time;

  
  if(time_since_colision > time_to_stand_still + time_to_drive_back){

    
    float diff = iRDistLeftNotLin - iRDistRightNotLin;
  
    float val = -diff/5;
  
    //Serial.println(val);
  
    rmspeed = 150 + val;
    lmspeed = 150 -val;
  
    int th = 200;
  
    if(rmspeed > th) rmspeed = th;
    if(lmspeed > th) lmspeed = th;
  
    if(rmspeed < -th) rmspeed = -th;
    if(lmspeed < -th) lmspeed = -th;
  
    //Serial.print("diff: ");
    //Serial.println(diff);
  
    if(iRDistMiddleNotLin > 200) {
      //rmspeed = 0;
      //lmspeed = 0;
  
      //rmspeed = -140;
      //lmspeed = -180;
    }

    forward = true;
  
    //serialDisplaySpeed();
  }
  else {
    // stand still for one second after colision
    // afterwards drive backwards with a tilt to the right

    //Serial.println("recovering");

    if(time_since_colision < time_to_stand_still){
      rmspeed = 0;
      lmspeed = 0;
    }
    else if(time_since_colision < time_to_stand_still + time_to_drive_back){
      int val = 1;
      if(forward == false) {
        val = 1;
      }
      else {
        val = -1;
      }
      rmspeed = -140*val;
      lmspeed = -190*val;
    }


  }
  
}


int time_to_turn_cor = 700;
int turn_start_time = 0;
int current_turn_time = 0;

bool turning = false;

float current_turn_val = 0;
myFloat adjustSpeeds(const unsigned short &avg_loop_lenght){


    float diff = iRDistLeftNotLin - iRDistRightNotLin;
  
    float val = (-diff/5);

  // if the loop takes longer be more carefull and stop earlier
  // a smaller value lets the robot turn earlier
  unsigned int front_threshold = 365 - avg_loop_lenght;

  if(turning == false && iRDistMiddleNotLin > front_threshold) {
      turning = true;
      turn_start_time = current_time;
      current_turn_val = val;
  }

  
    rmspeed = 150 + val;
    lmspeed = 150 -val;

    //Serial.println(val);
    
  if(turning == true) {
      if(current_turn_val < 0){
        lmspeed = 170;
        rmspeed = -170;
      } else {
        rmspeed = 170;
        lmspeed = -170;
      }
    }

    current_turn_time = current_time - turn_start_time;

    if(current_turn_time > time_to_turn_cor){
      if(iRDistMiddleNotLin > front_threshold){
        turning = true;
        turn_start_time = current_time;
      } else {
        turning = false;
      }
      
    }

    int th = 220;
  
    if(rmspeed > th) rmspeed = th;
    if(lmspeed > th) lmspeed = th;
  
    if(rmspeed < -th) rmspeed = -th;
    if(lmspeed < -th) lmspeed = -th;

    return val;
}

/*void getCollisionDirection(myFLoat x, myFloat y, myFloat z, myFLoat &x_out, myFloat &y_out, myFloat &z_out);
  x_out = -x;
  y_out = -y;
  z_out = -z:
}*/

myFloat valDiff = 0.0;
int r = 1;

/********************************************************************************
 * Main driving routine.
 ********************************************************************************/
void corridorBehaviour2(bool collectData, const unsigned long &avg_loop_lenght){

 int time_to_drive_on_after_collision = 100;
 time_to_stand_still = 1000;
 time_to_drive_back = 1000;
 time_to_turn = 600;



time_since_colision = current_time - colision_time;

Serial.print("time since collision: ");
Serial.println(time_since_colision);

  //current_time = millis();
  if(colided == true && time_since_colision > time_to_stand_still){
    //colision_time = millis();
    colided = false;
    forward = !forward;

      //calcMaxVec();

      //Serial.println("maxVec");
      //print_matrix(1,3, max_vec);

      
      r = random(1, 3); // either 1 or 2

      Serial.println("Collision----------------------------------------------");

  }

  if(collectData == false) time_to_drive_on_after_collision = 0;

    Serial.print("forward: ");
    Serial.println(forward);


    // normal driving behaviour with no recent collision
    if(time_since_colision > time_to_stand_still + time_to_drive_back + time_to_turn){
  
      // with the infrared sensors

        if(forward == true){
          valDiff = adjustSpeeds(avg_loop_lenght);
        }
      

      //serialDisplaySpeed();
    }
    // resolve collision
    else {
      // stand still for one second after colision
      // afterwards drive backwards with a tilt to either right or left
  
      Serial.println("recovering");
  
      if(time_since_colision < time_to_drive_on_after_collision){
        // in case we are currently collecting data, we let the robot drive on to generate better trainings data
        if(collectData == false){
          lmspeed = 0;
          rmspeed = 0;
        }

        //Serial.println("stand still");
      } else if(time_since_colision < time_to_stand_still + time_to_drive_on_after_collision){
        lmspeed = 0;
        rmspeed = 0;
      }
      // drive back
      else if(time_since_colision < time_to_stand_still + time_to_drive_back + time_to_drive_on_after_collision){
        Serial.println("drive back");

        if(forward == false){
          /*if(getVal(0, max_vec) < 0){
            rmspeed = -180;
            lmspeed = -140;
          } else {
            rmspeed = -140;
            lmspeed = -180;
          }*/

            if(r == 1){
              rmspeed = -150;
              lmspeed = -160;
            } else {
              lmspeed = -160;
              rmspeed = -150;
            }

          
        }else {
            //valDiff = adjustSpeeds();
            
            if(r == 1){
              rmspeed = 160;
              lmspeed = 150;
            } else {
              lmspeed = 150;
              rmspeed = 160;
            }
        }
      }
        else if(time_since_colision < time_to_stand_still + time_to_drive_back + time_to_turn + time_to_drive_on_after_collision){
         Serial.println("turn");
          
          if(forward == false){
            /*if(getVal(0, max_vec) < 0){
              rmspeed = -210;
              lmspeed = 0;
            } else {
              rmspeed = 0;
              lmspeed = -210;
            }*/

            if(r == 1){
              rmspeed = 170;
              lmspeed = -170;
            } else {
              rmspeed = -170;
              lmspeed = 170;
            }
            
          }else {
            //valDiff = adjustSpeeds();
            
            if(r == 1){
              rmspeed = 170;
              lmspeed = -170;
            } else {
              rmspeed = -170;
              lmspeed = 170;
            }
            
          }
        }


/*
        myFloat x = getVal(0, max_vec);
        myFloat y = getVal(1, max_vec);
        myFloat z = getVal(2, max_vec);

        int v = 1;
        if(y  > 0){
          v = -1;
        }

        lmspeed = 140 * v - 80 *x*v;
        rmspeed = 140 * v + 80 *x*v;

        if(lmspeed < -250) lmspeed = -250;
        if(lmspeed > 250) lmspeed = 250;
        if(rmspeed < -250) rmspeed = -250;
        if(rmspeed > 250) rmspeed = 250;

        Serial.print("lmspeed " );
        Serial.print(lmspeed);
        Serial.print(" " );
        Serial.print("rmspeed " );
        Serial.println(rmspeed);

        
        
      }*/
    }

    if(time_since_colision > time_to_stand_still + time_to_drive_back + time_to_turn) forward = true;
  
}



void corridorBehaviour3(bool collectData){

 int time_to_acc_after_collision = 1;
 time_to_stand_still = 200;
 time_to_drive_back = 1500;
 time_to_turn = 500;



  //current_time = millis();
  if(colided == true){
    //colision_time = millis();
    colided = false;
    forward = !forward;

      calcMaxVec();

      Serial.println("maxVec");
      print_matrix(1,3, max_vec);
  }

  time_since_colision = current_time - colision_time;


  float start = 0.8;
  float acc_mult = (float)(time_since_colision/(float)time_to_acc_after_collision)*(1.-start) + start;

  if(acc_mult > 1.0) acc_mult = 1.0;


    if(time_since_colision > time_to_stand_still + time_to_drive_back + time_to_turn){
  
      // with the infrared sensors

        if(forward == true){
          valDiff = adjustSpeeds(20);
        }
      

      //serialDisplaySpeed();
    }
    else {
      // stand still for one second after colision
      // afterwards drive backwards with a tilt to the right
  
      //Serial.println("recovering");
  
      if(time_since_colision < time_to_stand_still){
        if(collectData == false){
          lmspeed = 0;
          rmspeed = 0;
        }
      }
      else if(time_since_colision < time_to_stand_still + time_to_drive_back){

        if(forward == false){
          if(getVal(0, max_vec) < 0){
            rmspeed = -190;
            lmspeed = -140;
          } else if(getVal(0, max_vec) == 0) {
            rmspeed = -170;
            lmspeed = -170;
          } else {
            rmspeed = -140;
            lmspeed = -190;
          }
        } else {
            valDiff = adjustSpeeds(20);
        }
      }
        else if(time_since_colision < time_to_stand_still + time_to_drive_back + time_to_turn){
          if(forward == false){
            if(getVal(0, max_vec) < 0){
              rmspeed = -210;
              lmspeed = 0;
            } else {
              rmspeed = 0;
              lmspeed = -210;
            }
          } else {
            valDiff = adjustSpeeds(20);
          }
        }


/*
        myFloat x = getVal(0, max_vec);
        myFloat y = getVal(1, max_vec);
        myFloat z = getVal(2, max_vec);

        int v = 1;
        if(y  > 0){
          v = -1;
        }

        lmspeed = 140 * v - 80 *x*v;
        rmspeed = 140 * v + 80 *x*v;

        if(lmspeed < -250) lmspeed = -250;
        if(lmspeed > 250) lmspeed = 250;
        if(rmspeed < -250) rmspeed = -250;
        if(rmspeed > 250) rmspeed = 250;

        Serial.print("lmspeed " );
        Serial.print(lmspeed);
        Serial.print(" " );
        Serial.print("rmspeed " );
        Serial.println(rmspeed);

        
        
      }*/
    }

    lmspeed = lmspeed*acc_mult;
    rmspeed = rmspeed*acc_mult;

    if(time_since_colision > time_to_stand_still + time_to_drive_back + time_to_turn) forward = true;
  
}


unsigned long driveOnAfterCollision = 500;
unsigned long driveBackAfterCollision = 1100;

void forwardAndBackward(bool collectData, int val){

  //current_time = millis();
  if(colided == true){
    //colision_time = millis();
    colided = false;
    forward = !forward;
  }

  time_since_colision = current_time - colision_time;

  /*Serial.print(time_since_colision);
  Serial.print(" > ");
  Serial.println(driveOnAfterCollision + driveBackAfterCollision);
  */

    if(collectData == false){
      driveOnAfterCollision = 0;
    }
  

  
  if(time_since_colision > driveOnAfterCollision + driveBackAfterCollision){
  
    rmspeed = 160 * val;
    lmspeed = 160 * val; 

    forward = true;
  
  }
  else {

    if(time_since_colision < driveOnAfterCollision){
      rmspeed = 160 * val;
      lmspeed = 160 * val; 
    }
    else if(time_since_colision < driveOnAfterCollision + driveBackAfterCollision){
      rmspeed = -160 * val;
      lmspeed = -160 * val;
    }

  }
  
}

void forwardAfterCollision(){

  //current_time = millis();
  if(colided == true){
    //colision_time = millis();
    colided = false;
    forward = !forward;
  }
  int val = 1;

  time_since_colision = current_time - colision_time;

  /*Serial.print(time_since_colision);
  Serial.print(" > ");
  Serial.println(driveOnAfterCollision + driveBackAfterCollision);
  */

  
  if(time_since_colision > driveOnAfterCollision + driveBackAfterCollision){
  
    rmspeed = 160 * val;
    lmspeed = 160 * val; 

    forward = true;
  
  }
  else {

    if(time_since_colision < driveOnAfterCollision){
      rmspeed = 160 * val;
      lmspeed = 160 * val; 
    }
    else if(time_since_colision < driveOnAfterCollision + 0){
      rmspeed = -160 * val;
      lmspeed = -160 * val;
    }

  }
  
}


void setupBehaviour(){
  
}

void driveForwardTillCollision(){

  float val = 2.0;

  // collision detected!
  if(abs(accY) > val || abs(accX) > val || abs(accZ) > val){
    if(currentDrivingMode == FORWARD) switchToDrivingMode(BACKWARD,1000);
    else if(currentDrivingMode == BACKWARD) switchToDrivingMode(FORWARD,1000);
  }

  switchToNextDrivingMode();
  drive();
}

void driveRectangle(){
  if(currentDrivingMode == FORWARD && timeInCurrentMode() > 1000) switchToDrivingMode(LEFT,2000);

  switchToNextDrivingMode();
  drive();
}


unsigned long forward_t = 1000;
unsigned long still_t = 0;
unsigned long curr_t = 0;
int forW = -1;
int oforW = -1;

bool changed = false;

void driveForwardAndBackward(){

  unsigned long passedTime = millis() - curr_t;
  if(passedTime < forward_t){

  }
  else if(passedTime < forward_t + still_t){
    changed = true;
    forW = 0;
    
  } else {
    changed = true;
    curr_t = millis();

    
    forW = -oforW;
    oforW = -oforW;
    
  }
    rmspeed = forW*160;
    lmspeed = forW*160;


  if(changed) {
    //curr_t = millis();
    //changed = false;
  }
}

void driveForward(){
  if(colided == true){
    //colision_time = millis();
    colided = false;
    lmspeed = 0;
    rmspeed = 0;
  }

  currentDrivingMode = FORWARD;
  drive();
}


void driveForwardAndBreak(){
  if(currentDrivingMode == FORWARD && timeInCurrentMode() > 2000) switchToDrivingMode(STILL,1000);

  switchToNextDrivingMode();
  drive();
}


