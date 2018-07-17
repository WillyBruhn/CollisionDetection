#include "display.h"
#include "driveBehaviour.h"

#include "acceleration.h"

// Connect via i2c, default address #0 (A0-A2 not jumpered)
Adafruit_LiquidCrystal lcd(0);


void lcdDisplayMode(int line){
  lcd.setCursor(0, line);
  lcd.print("mode: ");

  lcd.setCursor(6, line);

  switch(currentMode){
    case NOTHING : lcd.print("NOTHING"); break;
    case DRIVING : lcd.print("DRIVING");
                   lcd.setCursor(14, line);
                   if(RUN == true) lcd.print("ON");
                   else lcd.print("OFF");
                   break;
    case MAXSPEED : lcd.print("MAXSPEED"); break;
  }

 // millis();
}

void serialDisplaySpeed(){
    Serial.print("lmspeed = ");
    Serial.print(lmspeed);
    Serial.print(", rmspeed = ");
    Serial.println(rmspeed);
}

// display the iR in given line
void displayInfraredDistances(int line){
  lcd.setCursor(0, line);
  lcd.print("IR: ");

  lcd.setCursor(4, line);
  lcd.print(iRDistLeft,0);

  lcd.setCursor(9, line);
  lcd.print(iRDistMiddle,0);

  lcd.setCursor(14, line);
  lcd.print(iRDistRight,0);

}

void displayMotor(int line){
  lcd.setCursor(0, line);
  lcd.print("motors: ");
  
  lcd.setCursor(10, line);
  lcd.print(lmspeed);

  lcd.setCursor(15, line);
  lcd.print(rmspeed);
}

void displayStatus(){
  lcdDisplayMode(0);
  
  //displayUltrasonicSensors(1);  // display in line 0

  printAccSenStatus(1);

  //displayInfraredDistances(2); // display in line 1

  displayInfraredDistancesNotLin(2);

  displayMotor(3);  // display in line 2
}

void displayInfraredDistancesNotLin(int line){
  lcd.setCursor(0, line);
  lcd.print("IR: ");

  lcd.setCursor(4, line);
  lcd.print(iRDistLeftNotLin,0);

  lcd.setCursor(9, line);
  lcd.print(iRDistMiddleNotLin,0);

  lcd.setCursor(14, line);
  lcd.print(iRDistRightNotLin,0);

}
