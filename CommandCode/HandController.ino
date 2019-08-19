#include <Servo.h>
// STD libs
#include <stdarg.h>
#include <stdio.h>

#include <boarddefs.h>
// Ardruino libs
// #include <IRremote.h>
// #include <IRremoteInt.h>
// #include <ir_Lego_PF_BitStreamEncoder.h>

// PIN 12 and Pin 13
// #define LED_PIN 12
// #define RECV_PIN 13

#define FINGER_COUNT 5

enum Finger{
    THUMB = 0,
    INDEX = 1,
    MIDDLE = 2,
    RING = 3,
    PINKY = 4
}

static Servo finger_servos[5];
static float finger_zero_state_degs[FINGER_COUNT] = {0, 0, 0, 180, 180};
static float finger_max_degs[FINGER_COUNT] = {180, 180, 180, 0, 0};
static float finger_min_degs[FINGER_COUNT] = {0, 0, 0, 180, 180};

static bool current_finger_states[FINGER_COUNT] = {0};

// single-finger control
void finger_control(Finger finger, int degree) {
    //enum Finger are int
    finger_servos[finger].write(adjusted_deg);
}

// multi-finger control
void multi_finger_control(int[] fingers_states) {
    float adjusted; int adjusted_deg;

    for (int i = 0; i < FINGER_COUNT; i++) {
        adjusted = ((finger_max_degs[i] - finger_min_degs[i]) / 180.0) * fingers_states[i] + finger_min_degs[i];
        adjusted_deg = (int) (adjusted + 0.5);
        finger_control((Finger)i, adjusted_deg);
    } 
}

void gesture_rock(){
    int finger_states_rock[FINGER_COUNT] = {0, 180, 180, 180, 180};
    multi_finger_control(finger_states_rock);
    //thumb first and then other four fingers
    delay(200);
    finger_states_rock = {0};
    multi_finger_control(finger_states_rock);
}

void gesture_scissor(){
    int finger_states_scissor[5] = {0, 180, 180, 0, 0};
    multi_finger_control(finger_states_scissor);
}

void gesture_paper(){
    int finger_states_paper[5] = {0};
    multi_finger_control(finger_states_paper);
}

// void on_received_remote_control(){
//     int finger_states[FINGER_COUNT];
//      // Don't read unless there you know there is data
//     if (irrecv.decode(&results)) {
//         HEXcode = results.value, HEX;

//         switch(HEXcode) {
//             case 2295: // Move Thumb
//                 finger_states = {180, 0, 0, 180, 180};
//                 multi_finger_control(finger_states);
//                 break;
//             case 34935: // Move Index
//                 finger_states = {0, 180, 0, 180, 180};
//                 multi_finger_control(finger_states);
//                 break;
//             case 18615: // Move Middle
//                 finger_states = {0, 0, 180, 180, 180};
//                 multi_finger_control(finger_states);
//                 break;
//             case 10455: // Move Ring
//                 finger_states = {0, 0, 0, 0, 180};
//                 multi_finger_control(finger_states);
//                 break;
//             case 43095: // Move Pinky
//                 finger_states = {0, 0, 0, 180, 0};
//                 multi_finger_control(finger_states);
//                 break;
//             case 26775: // Move 轮指
//                 finger_states = finger_zero_state_degs;
//                 multi_finger_control(finger_states);
//                 delay(1000);
//                 for (int i = 0; i < FINGER_COUNT * 2; i++) {
//                     int finger_index = i % 4;
//                     finger_states[finger_index] = abs(finger_states[finger_index] - 180);
//                     multi_finger_control(finger_states);
//                     delay(500); 
//                 }
//                 break;
//             case 6375: // Move Scissor
//                 gesture_scissor();
//                 break;
//             case 39015: // Move Paper
//                 gesture_paper();
//                 break; 
//             case 22695: // Move Rock
//                 gesture_rock();
//                 break;
//             case 8415: // Move intialize
//                 gesture_paper();
//                 break;
//             default: // Error
//                 break;
//         }
//         irrecv.resume(); 
//     }
// }

void setup() {
    //corresponding servo for the fingers
    int finger_pins[5] = {5, 6, 9, 10, 11};

    for (int i = 0; i < FINGER_COUNT; i++) {
        finger_servos[i].attach(finger_pins[i]);
    }
    pinMode(RECV_PIN,OUTPUT);
    Serial.begin(9600);//connect to serial port, baud rate is 9600

    gesture_paper();
    Serial.println("The Hand is ready!!" ) ;
}

void loop() {

  
//   // release the fingers if they idle too long
//   if (millis()-time_release > 60000)
//   {
//     time_release=millis();
//     all_motor_zeroing();
//     Serial.println("Bending the fingers too long, Release!");
//   }


//   //communicate with python
//   rock_paper_scissor_game(); 


//   //control by IR remote
//   IR_remote_control();
}