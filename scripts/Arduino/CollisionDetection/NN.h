#ifndef NN_H
#define NN_H

#include <SoftwareSerial.h>
#include <math.h>

extern unsigned long start_time;
extern unsigned long end_time;

typedef double myFloat;

myFloat sigmoid(myFloat x);

myFloat sigmoidFast(myFloat x);

void test_sigmoid(myFloat x, myFloat y, myFloat epsilon = 0.001);

void test_sigmoid();

void start_timer();

unsigned long time_elapsed();

void benchmarkSigmoid1();
void getTime();


extern myFloat NNoutput;

extern myFloat max_vec[3];

//------------------------------------------------------------------------------------
//extern const int inputNeurons;
/*const int n_hidden_1 = 5;
const int n_hidden_2 = 5;
const int n_classes = 2;

//extern myFloat W[inputNeurons][hiddenLayerSize];
//myFloat X[one_][inputNeurons];

myFloat X[1][inputNeurons];

myFloat h1[inputNeurons][n_hidden_1];
myFloat h2[n_hidden_1][n_hidden_2];
myFloat h_out[n_hidden_2][n_classes];

myFloat b1[n_hidden_1];
myFloat b2[n_hidden_2];
myFloat out[n_classes];


extern int currentXInd;
extern myFloat currentNNOutput;
*/

myFloat getVal(int ind, myFloat *arr);

void fillArr(myFloat* arr, int rows, int cols);

void print_matrix(unsigned int rows, unsigned int cols, myFloat *arr);

void matrix_mult(unsigned int rows1, unsigned int cols1, unsigned int cols2, myFloat *m1, myFloat *m2, myFloat *res);

void vector_add(unsigned int cols1, myFloat *m1, myFloat *m2, myFloat *res);

void sigmoid(unsigned int num, myFloat *V, myFloat *res);

myFloat evalSimpleNN(myFloat *X, myFloat *W);

bool setupNN();

void resetNN();

void getTime();

void start_timer();
unsigned long time_elapsed();

void shift_X(int times);

void feedNextXValue(int ind, myFloat accX, myFloat accY, myFloat accZ, myFloat gyroX, myFloat gyroY, myFloat gyroZ, int lmspeed , int rmspeed);

myFloat eval(void (* functionPointer)());

void testNN();

myFloat getNNThreshold() ;

void calcMaxVec();

#endif
