#include <SoftwareSerial.h>
#include <math.h>

#include "NN.h"
#include "model.h"

#if defined(ARDUINO) && ARDUINO >= 100
  #include "Arduino.h"
  #else
  #include "WProgram.h"
  #endif
#include <Wire.h>

extern SoftwareSerial sonars;

unsigned long start_time = 0;
unsigned long end_time;

myFloat sigmoid(myFloat x){
  // f(x) = x / (1 + abs(x))
  return(1.0/(1.0 + exp(-x)));
}

myFloat sigmoidFast(myFloat x){
  // f(x) = x / (1 + abs(x))
  return(x/(1+fabs(x)));
}

myFloat getNNThreshold() {
  return NN_Col_quotient;
}

// x argument
// y the correct value
void test_sigmoid(myFloat x, myFloat y, myFloat epsilon = 0.001){
  if(fabs(sigmoid(x) - y) >= epsilon) {
    Serial.print("f(");
    Serial.print(x);
    Serial.print(") = ");
    Serial.print(sigmoid(x));
    Serial.print(" != ");
    Serial.print(y);
    Serial.println("");
  }
}

void test_sigmoid(){
  test_sigmoid(0.9, 0.71);
  test_sigmoid(0.1, 0.52498);
  test_sigmoid(141, 1.0);
  test_sigmoid(0.0001, 0.5);
  test_sigmoid(0.01, 0.5025);
  test_sigmoid(-0.7, 0.3318);
}

myFloat getVal(int ind, myFloat *arr){
  return arr[ind];
}

myFloat myMax(int dim, myFloat *f){
  myFloat m = f[0];
  for(int i = 1; i < dim; i++){
    if(m < f[i]) m = f[i];
  }

  return m;
}


myFloat softMaxDim(int dim, myFloat *f, myFloat *res){
/*https://stackoverflow.com/questions/9906136/implementation-of-a-softmax-activation-function-for-neural-networks
 
  zj = wj . x + bj
  oj = exp(zj)/sum_i{ exp(zi) }
  log oj = zj - log sum_i{ exp(zi) }

  Let m be the max_i { zi } use the log-sum-exp trick:

  log oj = zj - log {sum_i { exp(zi + m - m)}}
   = zj - log {sum_i { exp(m) exp(zi - m) }},
   = zj - log {exp(m) sum_i {exp(zi - m)}}
   = zj - m - log {sum_i { exp(zi - m)}}

   oj = exp (zj - m - log{sum_i{exp(zi-m)}})*/

   /* myFloat sum = 0.0;

    for(int i = 0; i < dim; i++){
      sum += exp(f[i]);
    }

    for(int i = 0; i< dim; i++){
      res[i] = exp(f[i])/sum;
    }

    return res[0];*/

    myFloat m = myMax(dim,f);


    myFloat sum = 0.0;

    for(int i = 0; i < dim; i++){
      sum += exp(f[i] -m);
    }

    sum = log(sum);

    for(int j = 0; j< dim; j++){
      res[j] = exp(f[j] - m - sum);
    }

    return res[0];
}

myFloat softMaxDim_simple(int dim, myFloat *f, myFloat *res){

    myFloat sum = 0.0;

    for(int i = 0; i < dim; i++){
      sum += exp(f[i]);
    }

    for(int i = 0; i< dim; i++){
      res[i] = exp(f[i])/sum;
    }

    return res[0];

}

void start_timer(){
  start_time = millis();
}

unsigned long time_elapsed(){
  return(millis() - start_time);
}

void benchmarkSigmoid1(){
  for(myFloat i = -10; i < 10; i = i +0.01){
    sigmoid(i);
  }
}

void getTime(){
  Serial.println(time_elapsed());
  start_timer();
}

void matrix_mult(void (* functionPointer)(), unsigned int rows1, unsigned int cols1, unsigned int cols2, myFloat *m1, myFloat *m2, myFloat *res){

  int i, j, k;

      for (int i = 0; i < rows1; i++) {
        for (int j = 0; j < cols2; j++) {
            //res[i][j] = 0;
            res[i*cols2 + j] = 0;
            for (int k = 0; k < cols1; k++) {
                //c[i][j] += a[i][k] * b[k][j];
                res[i*cols2 + j] += m1[i*cols1 + k] * m2[k*cols2 + j];
            }

            functionPointer();
        }
    }
}

void vector_add(unsigned int cols1, myFloat *m1, myFloat *m2, myFloat *res){

  for (int i = 0; i < cols1; i++) {
    res[i] = m1[i] + m2[i];
  }
}

myFloat relu(myFloat x){
  if(x < 0) return 0;
  else return x;
}

myFloat relu(int size_, myFloat *arr_in, myFloat *arr_out){
  for(int i = 0; i < size_; i++){
    arr_out[i] = relu(arr_in[i]);
  }
}


myFloat normOfVector(myFloat x, myFloat y, myFloat z){
  return sqrt(x*x + y*y + z*z);
}

myFloat max_vec[3] = {0,0,0};

void calcMaxVec(){

  myFloat m = 0.0;
  myFloat d = 0.0;
  for(int i = 0; i < inputNeurons; i += 8){
    
    d = normOfVector(getVal(i, *X), getVal(i+1, *X), getVal(i+2, *X));

    Serial.print("d ");
    Serial.println(d);
    if(d > m) {
      setVar(0, getVal(i, *X)/d, max_vec);
      setVar(1, getVal(i+1, *X)/d, max_vec);
      setVar(2, getVal(i+2, *X)/d, max_vec);

      m = d;
    }

  }

}

//logisticRegression(functionPointer, *X, *h1, *layer1_raw, *b1, *layer1, *out);

myFloat logisticRegression(void (* functionPointer)(), myFloat *X,myFloat *h1,  myFloat *layer1_raw, myFloat *b1, myFloat *layer1, myFloat *out_raw, myFloat *out){
  matrix_mult(functionPointer, 1,inputNeurons,n_hidden_1,X,h1,layer1_raw);

  //Serial.println("layer1_raw: ");
  //print_matrix(1, n_classes, layer1_raw);

  vector_add(  n_hidden_1,  layer1_raw,  b1,  layer1);

  //Serial.println("layer1: ");
  //print_matrix(1, n_classes, layer1);

  //relu(2,  layer1, out_raw);
  softMaxDim(n_classes,  layer1, out);

  /*Serial.println("print in logisticRegression: ");

  print_matrix(1, inputNeurons, X);
  print_matrix(1, 2, layer1_raw);
  print_matrix(1,2,layer1);
  print_matrix(1,2,out);*/
  
  //softMaxDim(2,  out_raw, out);

  //out = out_raw;

  //Serial.println("out: ");
  //print_matrix(1, n_classes, out);
  return 1 - out[0];

  //return layer1[1];
}

myFloat evalNN(void (* functionPointer)(), myFloat *X, myFloat *h1, myFloat *layer1_raw, myFloat *layer1, myFloat *h2, myFloat *layer2_raw, myFloat *layer2, myFloat *h_out, myFloat *b1, myFloat *b2, myFloat *out_raw, myFloat *out, myFloat *outSmax){

  //myFloat layer1_raw[1][n_hidden_1];
  matrix_mult(functionPointer, 1,inputNeurons,n_hidden_1,X,h1,layer1_raw);

  //myFloat layer1[1][n_hidden_1];
  vector_add(  n_hidden_1,  layer1_raw,  b1,  layer1);

  //myFloat layer2_raw[1][n_hidden_2];
  matrix_mult(functionPointer, 1,n_hidden_1,n_hidden_2,layer1,h2,layer2_raw);

  //myFloat layer2[1][n_hidden_2];
  vector_add(  n_hidden_2,  layer2_raw,  b1,  layer2);

  //myFloat out_raw[1][n_classes];
  matrix_mult(functionPointer, 1,n_hidden_2,n_classes,layer2,h2,out_raw);

  vector_add(  n_classes, out_raw,  b1,  out);

  softMaxDim(2,  out, outSmax);

  return outSmax[1];

  //return out[0];
}

myFloat multilayer_perceptron_relu(void (* functionPointer)(), myFloat *X, myFloat *h1, myFloat *layer1_raw, myFloat *layer1, myFloat *h2, myFloat *layer2_raw, myFloat *layer2, myFloat *h_out, myFloat *b1, myFloat *b2, myFloat *out_raw, myFloat *out){

  //myFloat layer1_raw[1][n_hidden_1];
  matrix_mult(functionPointer, 1,inputNeurons,n_hidden_1,X,h1,layer1_raw);

  //myFloat layer1[1][n_hidden_1];
  vector_add(  n_hidden_1,  layer1_raw,  b1,  layer1);

  // first activation
  relu(n_hidden_1,  layer1, layer1_raw);  // yes I am reusing layer1_raw here!!!

  //print_matrix(1, n_hidden_1, layer1);
  //print_matrix(1, n_hidden_1, layer1_raw);
  //-------------------------------------------------------------------------------------------

  //myFloat layer2_raw[1][n_hidden_2];
  matrix_mult(functionPointer, 1,n_hidden_1,n_hidden_2,layer1_raw,h2,layer2_raw);

  //myFloat layer2[1][n_hidden_2];
  vector_add(  n_hidden_2,  layer2_raw,  b1,  layer2);

  // second activation
  relu(n_hidden_2,  layer2, layer2_raw);  // yes I am reusing layer2_raw here!!!

  //print_matrix(1, n_hidden_2, layer2_raw);
  //-------------------------------------------------------------------------------------------

  //myFloat out_raw[1][n_classes];
  matrix_mult(functionPointer, 1,n_hidden_2,n_classes,layer2_raw,h_out,out_raw);

  vector_add(  n_classes, out_raw,  b1,  out);

  //print_matrix(1,2, out);

  //relu(n_classes,  out, out_raw);
  //softMaxDim(n_classes,  out_raw, out);

  //relu(n_classes,  out, out_raw);
  //out = out_raw;


  softMaxDim(n_classes,  out, out_raw);
  //-------------------------------------------------------------------------------------------

  //out = out_raw;

  setVar(0, out_raw[0], out);
  setVar(1, out_raw[1], out);

  return out[1];
}


myFloat two_layer_perceptron(void (* functionPointer)(), myFloat *X, myFloat *h1, myFloat *layer1_raw, myFloat *layer1, myFloat *h_out, myFloat *b1, myFloat *out_raw, myFloat *out){

  //myFloat layer1_raw[1][n_hidden_1];
  matrix_mult(functionPointer, 1,inputNeurons,n_hidden_1,X,h1,layer1_raw);

  //myFloat layer1[1][n_hidden_1];
  vector_add(  n_hidden_1,  layer1_raw,  b1,  layer1);

  // first activation
  softMaxDim(n_hidden_1,  layer1, layer1_raw);  // yes I am reusing layer1_raw here!!!

  print_matrix(1, n_hidden_1, layer1);
  print_matrix(1, n_hidden_1, layer1_raw);
  //-------------------------------------------------------------------------------------------

  //myFloat out_raw[1][n_classes];
  matrix_mult(functionPointer, 1, n_hidden_1, n_classes, layer1_raw, h_out, out_raw);

  vector_add(  n_classes, out_raw,  b1,  out);

  //print_matrix(1,2, out);

  softMaxDim(n_classes,  out, out_raw);
  //-------------------------------------------------------------------------------------------

  return out_raw[1];

  //return out[0];
}

myFloat NNoutput = 0.0;

//------------------------------------------------------------------------------------

void fillArr(myFloat* arr, int rows, int cols) {
  int count = 0;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      // Multiply the current row by the number of columns to
      // get the appropriate offset.
      arr[i*cols + j] = 0;
      count++;
    }
  }
}

void reset(int num, myFloat *arr){
  for(int i = 0; i < num; i++){
    arr[i] = (myFloat) 0.0;
  }
}


void print_matrix(unsigned int rows, unsigned int cols, myFloat *arr){
  int i, j;
  for (i = 0; i < rows; i++) {
    for (j = 0; j < cols; j++) {
        Serial.print(arr[i*cols + j]);
        Serial.print(" ");
    }
    Serial.println("");
  }
  Serial.println("");
}


extern myFloat currentNNOutput = 0.0;
void resetNN(){
  reset(inputNeurons, *X);

  NNoutput = 0.0;

  currentNNOutput = 0.0;

  //print_matrix(1,60,*X);
}




void sigmoid(unsigned int num, myFloat *V, myFloat *res){
  for(unsigned int i = 0; i < num; i++){
    res[i] = sigmoid(V[i]);
  }
}

myFloat sum(int num, myFloat *v){
  myFloat sum = 0.0;
  for(int i = 0; i < num; i++){
    sum += v[i]; 
  }
  return sum;
}

/*myFloat evalSimpleNN(myFloat *X, myFloat *W){

  myFloat Between[1][1];
  matrix_mult(1,inputNeurons,1,X,W,*Between);

  myFloat s = sum(1, *Between);

  s += b;
  

  //myFloat Res[1][hiddenLayerSize];
  //sigmoid(hiddenLayerSize, *Between, *Res);

  //return *Res[0];

  return sigmoid(s);
}*/

void empty(){
  
}



bool testWrap(){
  eval(empty);
  
  Serial.print("Is: ");
  print_matrix(1,2, *out);

  Serial.print("Should: ");
  print_matrix(1,2, *y_small);

  double eps = 0.1;

  if(fabs(getVal(0, *out)-getVal(0, *y_small)) < eps && fabs(getVal(1, *out)-getVal(1, *y_small)) < eps) {
    Serial.println("Model seems good!");
  } else {
    Serial.println("Model has errors!");
   /* Serial.println(fabs(*out[0]-*y_small[0]));
    Serial.println(fabs(*out[1]-*y_small[1]));

    Serial.println(getVal(0, *out));
    Serial.println(getVal(1, *out));

    Serial.println(*y_small[0]);
    Serial.println(*y_small[1]);
    */

    return false;
  }

}

bool testNN2(){
  

  Serial.print("Testing model ");
  switch (model_choice){
    case 1:
      Serial.print("logistic_regression");
    break;
    case 2:
      Serial.print("two_layer_perceptron");
    break;
    case 3:
      Serial.print("multilayer_perceptron_relu");
    break;
  }
    
  Serial.println(" ...");
  test_case1();
  if(!testWrap()) return false;

  test_case2();
  if(!testWrap()) return false;

  test_case3();
  if(!testWrap()) return false;

  test_case4();
  if(!testWrap()) return false;

  
  return true;
}

bool setupNN() {

  //fillArr(*W, inputNeurons, hiddenLayerSize);


  //fillArr(*X, 1, inputNeurons);

  //return true;

  return testNN2();

}

void print_model(){
  Serial.println("h1");
  print_matrix(inputNeurons,n_hidden_1,*h1);


  Serial.println("h2");
  print_matrix(n_hidden_1,n_hidden_2,*h2);

  Serial.println("h_out");
  print_matrix(n_hidden_2,n_classes,*h_out);
}

void insertValInArrayAtPos(int i, myFloat val, myFloat *arr){

  /*Serial.print("Inserting ");
  Serial.print(val);
  Serial.print(" in ");
  Serial.println(i);*/
  arr[i] = val;
}


void shiftArrByX(int x, int size_, myFloat *arr){

  for(int i = size_-1 - x; i >= 0; i--){
    arr[i+x] = arr[i];
  }

}


void shift_X(int times){
  shiftArrByX(8*times,inputNeurons,*X);
}


int currentXInd = 0;

myFloat outTemp[1][n_classes] = {0,0};

short X_ind = 0;
void feedNextXValue(int ind, myFloat accX_, myFloat accY_, myFloat accZ_, myFloat gyroX_, myFloat gyroY_, myFloat gyroZ_, int lmspeed_ , int rmspeed_){

  double accNorm = 30.0;
  double motorNorm = 250.0;
  double gyroNorm = 200.0;

  //shiftArrByX(8,inputNeurons,*X);

  insertValInArrayAtPos(ind, (double)accX_/accNorm, *X);
  insertValInArrayAtPos(ind + 1, (double)accY_/accNorm, *X);
  insertValInArrayAtPos(ind + 2, (double)accZ_/accNorm, *X);
  insertValInArrayAtPos(ind + 3, (double)gyroX_/gyroNorm, *X);
  insertValInArrayAtPos(ind + 4, (double)gyroY_/gyroNorm, *X);
  insertValInArrayAtPos(ind + 5, (double)gyroZ_/gyroNorm, *X);
  insertValInArrayAtPos(ind + 6, (double)lmspeed_/motorNorm, *X);
  insertValInArrayAtPos(ind + 7, (double)rmspeed_/motorNorm, *X);
  

  // to make this work I need to redefine the matrix-multiplication
  // or make some array in the form of X_[0] = X[X_ind]; X_[1] = X[X_ind+1], ...
  /*insertValInArrayAtPos(X_ind, (double)accX_/accNorm, *X);
  insertValInArrayAtPos(X_ind + 1, (double)accY_/accNorm, *X);
  insertValInArrayAtPos(X_ind + 2, (double)accZ_/accNorm, *X);
  insertValInArrayAtPos(X_ind + 3, (double)gyroX_/gyroNorm, *X);
  insertValInArrayAtPos(X_ind + 4, (double)gyroY_/gyroNorm, *X);
  insertValInArrayAtPos(X_ind + 5, (double)gyroZ_/gyroNorm, *X);
  insertValInArrayAtPos(X_ind + 6, (double)lmspeed_/motorNorm, *X);
  insertValInArrayAtPos(X_ind + 7, (double)rmspeed_/motorNorm, *X);


  X_ind += 8;
  X_ind = X_ind % inputNeurons;*/


  /*Serial.print("Acc: ");
  Serial.print(accX_);
  Serial.print(" ");
  Serial.print(accY_);
  Serial.print(" ");
  Serial.print(accZ_);
  Serial.print(" ");
  Serial.print(", Gyro: ");
  Serial.print(gyroX_);
  Serial.print(" ");
  Serial.print(gyroY_);
  Serial.print(" ");
  Serial.print(gyroZ_);

  Serial.print(". AccNorm: ");
  Serial.print(accX_/accNorm);
  Serial.print(" ");
  Serial.print(accY_/accNorm);
  Serial.print(" ");
  Serial.print(accZ_/accNorm);
  Serial.print(" ");
  Serial.print(", GyroNorm: ");
  Serial.print(gyroX_/gyroNorm);
  Serial.print(" ");
  Serial.print(gyroY_/gyroNorm);
  Serial.print(" ");
  Serial.print(gyroZ_/gyroNorm);
  Serial.println(" ");

  /*Serial.print(". Motor: ");
  Serial.println(rmspeed_);
  Serial.print(". MotorNorm: ");
  Serial.println((double)rmspeed_/motorNorm);*/
}


myFloat eval(void (* functionPointer)()){
  //currentNNOutput = evalNN(functionPointer, *X, *h1,  *layer1_raw,  *layer1,  *h2,  *layer2_raw,  *layer2,  *h_out,  *b1,  *b2,  *out_raw,  *out, *outTemp);

  //currentNNOutput = multilayer_perceptron6(functionPointer, *X, *h1,  *layer1_raw,  *layer1,  *h2,  *layer2_raw,  *layer2,  *h_out,  *b1,  *b2,  *out_raw,  *out);

  switch (model_choice){
    case 1:
      currentNNOutput = logisticRegression(functionPointer, *X, *h1, *layer1_raw, *b1, *layer1, *out_raw, *out);
    break;
    
    case 2:
      currentNNOutput = two_layer_perceptron(functionPointer, *X, *h1,  *layer1_raw,  *layer1,  *h_out,  *b1,  *out_raw,  *out);
    break;
    
    case 3:
      currentNNOutput = multilayer_perceptron_relu(functionPointer, *X, *h1,  *layer1_raw,  *layer1,  *h2,  *layer2_raw,  *layer2,  *h_out,  *b1,  *b2,  *out_raw,  *out);
    break;
  }

  //print_matrix(1,2,*out_raw);

 
  return currentNNOutput;
}




//old
/*
myFloat feedNextXValue(myFloat accX, myFloat accY, myFloat accZ, int lmspeed , int rmspeed){

  if(currentXInd > inputNeurons){
    //currentNNOutput = evalSimpleNN(*X,*W);
    currentNNOutput = evalNN(*X, *h1,  *layer1_raw,  *layer1,  *h2,  *layer2_raw,  *layer2,  *h_out,  *b1,  *b2,  *out_raw,  *out);
    currentXInd = 0;




    currentNNOutput = softMaxDim(n_classes, *out, *outTemp);

    //print_matrix(1, 60, *X);
    //delay(10000);
    
  }
  
  //X[currentXInd] = accX;
  //*X[currentXInd + 1] = accY;
  //*X[currentXInd + 2] = accZ;

  myFloat accNorm = 30;
  myFloat motorNorm = 250;

  insertValInArrayAtPos(currentXInd, accX/accNorm, *X);
  insertValInArrayAtPos(currentXInd + 1, accY/accNorm, *X);
  insertValInArrayAtPos(currentXInd + 2, accZ/accNorm, *X);
  insertValInArrayAtPos(currentXInd + 3, lmspeed/motorNorm, *X);
  insertValInArrayAtPos(currentXInd + 4, rmspeed/motorNorm, *X);

  currentXInd += 5;


  return currentNNOutput;
}

*/



myFloat Q[1][4] = {1,2,3,1};

myFloat T[1][10] = {1,2,3,4,5,6,7,8,9,10};

myFloat P[1][4] = {1,2,3,1};

myFloat R[1][4] = {0,5,0,0};

myFloat M[4][4] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};

void testNN(myFloat *Q, myFloat *P, myFloat *M, myFloat *R){


  //vector_add(4, Q, P, R);
//matrix_mult(unsigned int rows1, unsigned int cols1, unsigned int cols2, myFloat *m1, myFloat *m2, myFloat *res)
  //matrix_mult(1, 4, 4, Q, M, R);

  Serial.println("Matrix");
  print_matrix(1,4,R);
  
}

void testNN(){
  //testNN(*Q,*P,*M,*R);

  //print_matrix(100,5,*h1);

  
  //myFloat a = softMaxDim(4, *Q, *P);

  

  //Serial.println(a);


  print_matrix(1,10,*T);
  shiftArrByX(1,10,*T);

  print_matrix(1,10,*T);

  shiftArrByX(5,10,*T);

  print_matrix(1,10,*T);

  
}



