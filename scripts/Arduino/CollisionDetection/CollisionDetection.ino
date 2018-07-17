char foo; // random code in the beginning needed
// see:
// https://electronics.stackexchange.com/questions/4990/why-am-i-receiving-error-serial-was-not-declared-in-this-scope-when-building
// for more details

#include <Wire.h>
#include <SoftwareSerial.h>


#include "distanceSensors.h"
#include "utility.h"
#include "modes.h"
#include "display.h"
#include "acceleration.h"
#include "NN.h"

#include "driveBehaviour.h"

extern Adafruit_LiquidCrystal lcd;

/********************************************************************************
 * Switch on/off the motor
 ********************************************************************************/
int interruption = 18;  //


// depending on where the button is it is either HIGH or LOW
volatile bool interrupted = false;    // better to be volatile. Otherwise the compiler might replace it with false, if it never gets changed
                                      // in the rest of the code

/********************************************************************************
 * Register a collision by buttonpress.
 ********************************************************************************/

int buttonPin = 40;

int buttonState = HIGH;
int oldButtonState = HIGH;


void receive_key_press();
void get_new_acc();


/********************************************************************************
 * A data_pack holds all the relevant information in a given time-poit for the 
 * NN. 
 ********************************************************************************/
struct data_pack{
  double accX;
  double accY;
  double accZ;

  double gyroX;
  double gyroY;
  double gyroZ;

  int lmspeed;
  int rmspeed;

  bool colided;

  float colided_prediction;

  unsigned long m_time;

  void set_values(double x, double y, double z, double gx, double gy, double gz, int l, int r, float colided_pred, bool col, unsigned long t){
    this->accX = x;
    this->accY = y;
    this->accZ = z;

    this->gyroX = gx;
    this->gyroY = gy;
    this->gyroZ = gz;

    this->lmspeed = l;
    this->rmspeed = r;

    this->colided = col;

    this->colided_prediction = colided_pred;
    
    this->m_time = t;
  }

  void print_values() const{
        
    Serial.print(this->m_time);
    Serial.print(" ");

    get_new_acc();
    
    Serial.print(this->accX);
    Serial.print(" ");
    Serial.print(this->accY);
    Serial.print(" ");
    Serial.print(this->accZ);
    Serial.print(" ");

    get_new_acc();

    Serial.print(this->gyroX);
    Serial.print(" ");
    Serial.print(this->gyroY);
    Serial.print(" ");
    Serial.print(this->gyroZ);
    Serial.print(" ");

    get_new_acc();
    
    Serial.print(this->lmspeed);
    Serial.print(" ");
    Serial.print(this->rmspeed);
    Serial.print(" ");

    get_new_acc();

    Serial.print(this->colided_prediction);
    Serial.print(" ");
    
    Serial.print(this->colided);

    Serial.println("");

  }
};


 /********************************************************************************
 * Circular buffer.
 * Buffers all the sensor-data.
 * One index for the end of the buffer.
 * One index for writing the data to the Serial-port.
 * One buffer for the ANN.
 ********************************************************************************/
class myBuffer{
private:

  unsigned short end_index = 0;
  unsigned short NN_index = 0;
  unsigned short write_index = 0;

  const static short m_size = 30;

  data_pack m_buffer[m_size];

public:
   /********************************************************************************
   * returns the next index of the circullar Buffer
   ********************************************************************************/
  short next_index(unsigned short &ind){
    if(ind < m_size-1) return ind+1;
    return 0;
  }

   /********************************************************************************
   * Method to insert all relevant values. The buffer is "full" if the next index 
   * at which we want to insert a new value is either end_index or NN_index. In this
   * case the certain index has to read the rest of the buffer first.
   ********************************************************************************/
  void push_new_values(double x, double y, double z, double gx, double gy, double gz, int l, int r, float col_pred, bool col, unsigned long t){
    if(NN_index != next_index(end_index) && write_index != next_index(end_index)){
      m_buffer[end_index].set_values(x,y,z,gx,gy,gz,l,r, col_pred, col ,t);
    }
    end_index = next_index(end_index);
  }

   /********************************************************************************
   * Specifies the distance between the two given indices in circular direction
   ********************************************************************************/
  int buffer_distance(int begin_ind, int end_ind){
    if(begin_ind <= end_ind) return end_ind - begin_ind;
    return m_size - (begin_ind - end_ind);
  }

   /********************************************************************************
   * First drops the last few values of the NN and afterwards inserts the new values
   * in the beginning.
   ********************************************************************************/
  void feed_values_to_NN(){

    shift_X(buffer_distance(NN_index,end_index));

    int i = 0;
    while(end_index != NN_index){
      feedNextXValue(i, m_buffer[NN_index].accX,  m_buffer[NN_index].accY,  m_buffer[NN_index].accZ, m_buffer[NN_index].gyroX,  m_buffer[NN_index].gyroY,  m_buffer[NN_index].gyroZ, m_buffer[NN_index].lmspeed, m_buffer[NN_index].rmspeed);
      NN_index = next_index(NN_index);
      i+=8;
      
      get_new_acc();
    }
  }

   /********************************************************************************
   * Writes the contents of the buffer (that have not been sent before) 
   * to the serial port.
   ********************************************************************************/
    void write_values_to_serial_port(){
      
      while(end_index != write_index){
        
        m_buffer[write_index].print_values();
        write_index = next_index(write_index);
        get_new_acc();
      }
    }

};

myBuffer Buffer;

volatile bool last_frame_collision = false; // indicates whether in the last loop occured a collision
unsigned long last_accTime = 0; // holds the timepoint (in ms) of the last time sensor data was collected 
unsigned cu_t = 0;

/********************************************************************************
 * This function is used to collect new sensor-data. When called, it is first 
 * checked, if enough time has passed. By calling this function as often as
 * possible a relatively stable frequency of incoming data is ensured.
 ********************************************************************************/
void get_new_acc(){
  cu_t = millis();
  if(cu_t - last_accTime >= 9){
    last_accTime = cu_t;
    readAccSenStatus();

    Buffer.push_new_values(accX, accY, accZ, gyroX, gyroY, gyroZ, lmspeed, rmspeed, NNoutput, last_frame_collision, cu_t);

    if(last_frame_collision == true) {
      last_frame_collision = false;
    }
  }
}

bool displayUpdated = false;  // the lcd-display displays, if a collision occured
bool displayCleared = false;


float NN_trigger = 0.0; // value with which the threshold of the NN was surpassed in the last loop

/********************************************************************************
 * Adjust this parameter for a more or less sensitve collision-prediction.
 ********************************************************************************/
const double NN_threshold = 0.3;  // <---------------------------------------------------------------------------------------


unsigned long avg_loop_length = 0;  // holds the average duration of a loop


/********************************************************************************
 * Upon collision this function can be called to display some stats about the collision
 * on the lcd-screen.
 * 
 * t_ms ... time in ms to display the collision on the screen
 ********************************************************************************/
void displayCollision(unsigned long col_time, int t_ms){
  get_new_acc();

 if(t_ms > millis() - col_time){
    if(displayUpdated == false){
      lcd.setCursor(0, 0);
      lcd.print("Collision detected!");
      lcd.setCursor(0, 1);
      lcd.print("NN :");
      lcd.setCursor(5, 1);
      lcd.print(NN_threshold);
      lcd.setCursor(9, 1);
      lcd.print("<");
      lcd.setCursor(11, 1);
      lcd.print(NN_trigger);
      lcd.setCursor(0, 2);
      lcd.print("loop length: ");
      lcd.setCursor(14, 2);
      lcd.print(avg_loop_length);
      displayUpdated = true;
      displayCleared = false;

      get_new_acc();
    }

  } else {
    if(displayCleared == false){
      lcd.clear();
      displayCleared = true;
      displayUpdated = false;

      get_new_acc();
    }
    
  }
 
}

/********************************************************************************
 * Select the right mode here.
 * DRIVE ... the robot drives autonomously and tries to detect collisions
 * COLLECTDATA ... stay conected with the robot and register a collision with 
 *                  a enter press. Used for collecting trainings-data
 ********************************************************************************/
enum collision_mode {DRIVE, COLLECTDATA, ADJUSTNNTHRESHOLD};
const collision_mode active_mode = ADJUSTNNTHRESHOLD;

enum drive_modes {FORWARD_BACKWARD, CORRIDOR, ONLY_FORWARD, ONLY_BACKWARD, BACKWARD_FORWARD, BACKWARD_FORWARD_PENDULUM};
const drive_modes drive_mode = CORRIDOR;

// During this time new collisions are ignored. Even if the NN sends a collision signal
// the robot continues driving. Only relevant for collision_mode = DRIVE
int time_to_ignore_collisions = 1500; 


unsigned long last_loop_begin = millis(); // used to calculate the duration of a loop

bool collectData = false;
void setup() {
  Serial.begin(2000000);
  Wire.begin();

  // robot is switched off in the beginning
  RUN = false;
  
  // set up the LCD's number of rows and columns: 
  lcd.begin(20, 4);                 

  lcd.setCursor(0, 0);
  lcd.print("Initializing...");
  
  delay(1000); 
  lcd.clear();

  attachInterrupt(digitalPinToInterrupt(interruption), interruptButton, RISING);

  for(unsigned int i = 0; i < frameSize; i++){
    iRVoltageRight[i] = 0;
    iRVoltageMiddle[i] = 0;
    iRVoltageLeft[i] = 0;
  }

  initAccSen();

  // automatically test the correctness of the model
  bool good = setupNN();
  if(!good) delay(1000);
  
  if(DRIVE != active_mode) collectData = true;

}

bool collision_prediction = false;

void loop() {

  unsigned long c_t = millis();

  // this converges against the average loop-length over time
  avg_loop_length = (avg_loop_length*9 + (c_t - last_loop_begin))/10;
  last_loop_begin = c_t;

  get_new_acc();
  // check if interruptbutton was pressed
  if(interrupted){
    interrupted = false;

    if(currentMode == DRIVING){
      stopTREX();

      RUN = !RUN;

      if(RUN == true){
        currentDrivingMode = FORWARD;
        nextDrivingMode = FORWARD;
      }

        // for training collisions from the stand 
        if(RUN == true && (drive_mode == ONLY_FORWARD || drive_mode == ONLY_BACKWARD)){
          colided = true;
          last_frame_collision = true;
          colision_time = millis();
        }
    }

    // NN
    resetNN();

  }

  get_new_acc();
    
    buttonState = digitalRead(buttonPin);

    receive_key_press();
    
    if (buttonState == HIGH) {
      get_new_acc();
  
      if(oldButtonState != buttonState) {
        colided = true;
        last_frame_collision = true;
        colision_time = millis();

        //Serial.println("Collision");
      }
    }
    oldButtonState = buttonState;

    get_new_acc();
  
    if(currentMode != NOTHING){

      get_new_acc();
  
      getInfraredNotLin();
  
      get_new_acc();
    }

    unsigned long t1 = millis();
   

    get_new_acc();

    if(DRIVE == active_mode){
      Serial.println(NNoutput);
    }

    if(DRIVE != active_mode) {
      Buffer.write_values_to_serial_port();
    }

    if(currentMode == DRIVING && RUN == true){
      get_new_acc();    // 0

      current_time = millis();
      
      switch(drive_mode){
        case ONLY_FORWARD:
          lmspeed = 160;
          rmspeed = 160;
        break;
        
        case FORWARD_BACKWARD:
          forwardAndBackward(collectData, 1); // <-------------------------
        break;

        case BACKWARD_FORWARD:
          forwardAndBackward(collectData, -1);
        break;

        case CORRIDOR:
          corridorBehaviour2(collectData, avg_loop_length);
          //corridorBehaviour3(collectData);
        break;

        case BACKWARD_FORWARD_PENDULUM:
          driveForwardAndBackward();
        break;

        case ONLY_BACKWARD:
          lmspeed = -160;
          rmspeed = -160;
        break;
        
      }

      get_new_acc(); // 1

  
      if(DRIVE == active_mode){
        //if(NNoutput > getNNThreshold()){
        if(NNoutput >= NN_threshold && time_to_ignore_collisions < millis() - colision_time){
          NN_trigger = NNoutput;
          
        
          Serial.println("collision detected!");

          colided = true;
          last_frame_collision = true;
          colision_time = millis();
        }
      }
  
      get_new_acc();
      MasterSend(startbyte,2,lmspeed,lmbrake,rmspeed,rmbrake,sv[0],sv[1],sv[2],sv[3],sv[4],sv[5],devibrate,sensitivity,lowbat,i2caddr,i2cfreq);    
      get_new_acc();
      
    }
  
    if(true){
      Buffer.feed_values_to_NN();
      NNoutput = eval(get_new_acc);
    }

    if(DRIVE == active_mode) displayCollision(colision_time, 5000);
}

void interruptButton() {
  //Serial.println("Interrupt detected ...");
  interrupted = true;
}

void stopTREX(){
  //Serial.println("stopTREX()...");
  lmspeed = 0;
  rmspeed = 0;
  
  MasterSend(startbyte,2,lmspeed,lmbrake,rmspeed,rmbrake,sv[0],sv[1],sv[2],sv[3],sv[4],sv[5],devibrate,sensitivity,lowbat,i2caddr,i2cfreq);

}

/********************************************************************************
 * Function to register collisions
 ********************************************************************************/
void receive_key_press(){

  if(Serial.available() > 0){

    int inc_byte = Serial.read();
  
    if(inc_byte == 107){
      //Serial.println("k pressed");
      buttonState = HIGH;
    }
  
    if(inc_byte == 108){
      //Serial.println("l pressed");
      interruptButton();
    }
  }
}

