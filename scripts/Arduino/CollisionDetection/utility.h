#ifndef UTILITY_H
#define UTILITY_H

//#include <SoftwareSerial.h>

double getAverage(float arr[], int size);

double getMin(float arr[], int size);

// shifts all elements by one
// adds the new element in front and discards the last one
// better do this with linked list maybe?
void appendToArray(float arr[], int size, float val);

void printArray2Console(float arr[], int size);
#endif
