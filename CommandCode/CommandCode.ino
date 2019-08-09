#include <Servo.h>
#include <stdarg.h>
#include <stdio.h>
#include <boarddefs.h>

#include <IRremote.h>
#include <IRremoteInt.h>
#include <ir_Lego_PF_BitStreamEncoder.h>

int LED_pin=12;
int RECV_pin=13;
IRrecv irrecv(RECV_pin); // Initiate IR signal input
decode_results results; // Save signal structure
word HEXcode;  //HEXcode for IR remote control
unsigned long time_release;


Servo finger_servo[5];

int finger_pin[5]={5,6,9,10,11};   //corresponding servo for the fingers
int finger_zero_state_angle[]={0,0,0,180,180};
int finger_state[5]={1,1,1,1,1}; // 1 is held out the finger

char serialData;
enum gesture{paper ='0', rock = '1', scissor = '2'};



//================================================================================================
void  multi_motor_control(int finger[], int deg[] ,int num_finger )
{
    for(int f=0; f<num_finger; f++)
    { 
      int finger_idx=finger[f]-1;
      
      finger_servo[ finger_idx ].write(deg[ f ]);
    }    
  
  // state_update
  for(int f=0; f<num_finger; f++)
  {      
     if ( deg[f]== finger_zero_state_angle[finger[f]-1] )
     {
        finger_state[finger[f]-1]=1;
     }
     else
     {
        finger_state[finger[f]-1]=0;
     }
  }
  state_print();
}
//================================================================================================
void all_motor_zeroing()
{
  int temp_finger[5] = {1,2,3,4,5};
  int temp_state[5] = {1,1,1,1,1};
  multi_motor_state_control(temp_finger, temp_state ,5);
}
//================================================================================================
int state2deg(int finger, int state)
{
  //if state == 0, then finger down
  
  if(state==0)
  {
    return abs(180-finger_zero_state_angle[finger-1]);
  }

  else if (state==1)
  {
   // if state == 1, then finger up
    return finger_zero_state_angle[finger-1];
  }

  else
  {
   Serial.println("state2deg error");
  }
}
//================================================================================================

void finger_trig(int finger[], int num_finger )
{ 
  //control one finger
  int finger_state_temp[5];
  for(int f=0;f<num_finger; f++)
  {
    finger_state_temp[f]=abs(finger_state[finger[f]-1]-1);   //change to opposite state
  }
  
  multi_motor_state_control(finger,finger_state_temp ,num_finger );
}
//================================================================================================
void  multi_motor_state_control(int finger[], int state[] ,int num_finger )
{
  //control multiple fingers
  int deg[5];
  for(int f=0;f<num_finger; f++)
  {
    deg[f]=state2deg(finger[f], state[f]);
  }
  
  multi_motor_control(finger, deg ,num_finger );
}
//================================================================================================
void state_print()
{
  String state_show;
  for(int i=0;i<=4;i++)
  {
    if (finger_state[i]==1)
    {
      state_show=state_show+"|| ";
    }
    else 
    {
      state_show=state_show+"_ ";
    }
  }

  Serial.println(state_show);
}

//================================================================================================

void gesture_paper(){
  all_motor_zeroing();   
}
//================================================================================================

void gesture_scissor(){
  int temp_finger[5] = {1,2,3,4,5};
  int temp_state[5] = {0,1,1,0,0};
  multi_motor_state_control(temp_finger, temp_state ,5);     
}
//================================================================================================

void gesture_rock(){
  int temp_finger_2[1] = {1};
  int temp_state_2[1] = {0};
  multi_motor_state_control(temp_finger_2, temp_state_2 ,1 );
  delay(200);
  int temp_finger[4] = {2,3,4,5};
  int temp_state[4] = {0,0,0,0};
  multi_motor_state_control(temp_finger, temp_state ,4 );
}



void rock_paper_scissor_game() {
  
 if(Serial.available() > 0 ){
  // if receive data
  
    serialData = Serial.read();
    
    if(serialData == paper){
        gesture_paper();
       
    }else if (serialData == rock) {       
        gesture_rock();
        
    }else if(serialData = scissor){
        gesture_scissor();
    }

  }
}


void IR_remote_control(){
  // command for IR remote mode
  
 if (irrecv.decode(&results)) // Don't read unless there you know there is data
  {
      time_release=millis();
      //Serial.println(results.value, HEX);
      HEXcode= results.value, HEX;
      //Serial.println(HEXcode);
  
      
 //------1 : 2295 -----------------------------
      if (HEXcode == 2295)
      {
        // move thumb
        int temp[]={1};
        finger_trig(temp, 1);
      }
 //------ 2 : 34935-----------------------------
      else if (HEXcode == 34935)
      {
        // move index finger
        int temp[]={2};
        finger_trig(temp, 1 );
      }
      
 //------ 3 : 18615-----------------------------
      else if(HEXcode == 18615) 
      {
        // move middle finger
        int temp[]={3};
        finger_trig(temp, 1 );
      }
 //------ 4 : 10455-----------------------------
      else if(HEXcode==10455)
      {
        // move ring finger
        int temp[]={4};
        finger_trig(temp, 1 );
      }
 //------ 5 : 43095-----------------------------
      else if(HEXcode==43095)
      {
        // move little finger
        int temp[]={5};
        finger_trig(temp, 1 );
      }
 //------ 6 : 26775-----------------------------
      else if(HEXcode==26775)
      {
        all_motor_zeroing();   
        delay(1000);
        int temp[]={1};
        finger_trig(temp, 1 );
        delay(500);
        temp[0]={2};
        finger_trig(temp, 1 );
        delay(500);
        temp[0]={3};
        finger_trig(temp, 1 );
        delay(500);
        temp[0]={4};
        finger_trig(temp, 1 );
        delay(500);
        temp[0]={5};
        finger_trig(temp, 1 );
        delay(500);
        temp[0]={5};
        finger_trig(temp, 1 );
        delay(500);
        temp[0]={4};
        finger_trig(temp, 1 );
        delay(500);
        temp[0]={3};
        finger_trig(temp, 1 );
        delay(500);
        temp[0]={2};
        finger_trig(temp, 1 );
        delay(500);
        temp[0]={1};
        finger_trig(temp, 1 );
      }
 //------ 7 : 6375----------------------------- scissor
      else if(HEXcode==6375)
      {
        gesture_scissor();
      }
 //------ 8 : 39015----------------------------- paper
       else if(HEXcode==39015)
      {
        gesture_rock();
      }
 //------ 9 : 22695----------------------------- rock
      else if(HEXcode==22695)
      {
        gesture_paper();
      }
 //------ SETUP : 8415-----------------------------
 //all fingers held up 
      else if(HEXcode==8415)
      { 
         int temp_finger[5] = {1,2,3,4,5};
         int temp_state[5] = {1,1,1,1,1};
         multi_motor_state_control(temp_finger, temp_state ,5 );
     
      }
      

      irrecv.resume(); 
  }

  
}



//================================================================================================
//================================================================================================
void setup() {
  for (int i = 0; i<5; i++)
  {
    finger_servo[i].attach(finger_pin[i]);
  }
  pinMode(RECV_pin,OUTPUT);
    
  Serial.begin(9600);//connect to serial port, baud rate is 9600
  

  //zeroing the motors
  for( int f=0; f<5 ; f++)
  {
    finger_servo[ f ].write(finger_zero_state_angle[ f ]);
    
  }
  delay(10);

  irrecv.blink13(true); // if signal is received, then pin13 led light blink
  irrecv.enableIRIn(); // enable the singal receival function
  pinMode(LED_pin,OUTPUT);
  time_release=millis();

  digitalWrite(LED_pin,HIGH);
  digitalWrite(A0,HIGH);
  Serial.println("The Hand is ready!!" ) ;
  state_print();
  
  
}
//================================================================================================



void loop() 
{

  
  // release the fingers if they idle too long
  if (millis()-time_release > 60000)
  {
    time_release=millis();
    all_motor_zeroing();
    Serial.println("Bending the fingers too long, Release!");
  }


  //communicate with python
  rock_paper_scissor_game(); 


  //control by IR remote
  IR_remote_control();
  

  

}
