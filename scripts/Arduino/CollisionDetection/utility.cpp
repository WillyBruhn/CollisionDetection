#include <SoftwareSerial.h>

#if defined(ARDUINO) && ARDUINO >= 100
  #include "Arduino.h"
  #else
  #include "WProgram.h"
  #endif
#include <Wire.h>

#include "utility.h"

double getAverage(float arr[], int size) {

   int i;
   float avg;
   float sum = 0;

   for (i = 0; i < size; ++i) {
      sum += arr[i];
   }

   avg = sum / size;

   return avg;
}

double getMin(float arr[], int size) {

   int i;
   float minV = arr[0];

   for (i = 1; i < size; ++i) {
      if(minV > arr[i]) minV = arr[i];
   }

   return minV;
}


// shifts all elements by one
// adds the new element in front and discards the last one
// better do this with linked list maybe?
void appendToArray(float arr[], int size, float val){
  for(unsigned int i = size-1; i > 0; i--){
    arr[i] = arr[i-1];
  }

  arr[0] = val;
}

void printArray2Console(float arr[], int size){
  for(unsigned int i = 0; i < size; i++){
    Serial.print(arr[i]);
    Serial.print(";");
  }

  Serial.println("");
}
