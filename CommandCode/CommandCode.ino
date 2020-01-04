// Ardruino libs
#include <Servo.h>
#include <boarddefs.h>
#include <IRremote.h>
#include <IRremoteInt.h>
#include <ir_Lego_PF_BitStreamEncoder.h>

#define IR_RECV_PIN 13
#define FINGER_COUNT 5
#define IDLE_TIMEOUT 60000 // milliseconds
#define SERIAL_PORT 9600
#define MOVEMENT_WAITING_TIME 500 // milliseconds
#define THUMB_COLLISION_WAIT 200 // milliseconds
#define CAUSE_COLLISION_DEG 60
#define THUMB_COLLISION_MAX_PERCENTAGE 60
#define SERVO_MALFUNCTION_PREVENTION_DELAY 0 // milliseconds

#define ENABLE_IR_REMOTE_CONTROL 1
#define ENABLE_SERIAL_PORT_CONTROL 1

typedef enum {
    THUMB = 0,
    INDEX = 1,
    MIDDLE = 2,
    RING = 3,
    PINKY = 4
} Finger;

// IRRemote controller of Carmp3 kind
typedef enum {
    BTN_CH_MINUS = 41565,
    BTN_CH = 25245,
    BTN_CH_PLUS = 57885,
    BTN_PREV = 8925,
    BTN_NEXT = 765,
    BTN_PLAY = 49725,
    BTN_VOL_DOWN = 57375,
    BTN_VOL_UP = 43095,
    BTN_EQ = 36975,
    BTN_0 = 26775,
    BTN_100_PLUS = 39015,
    BTN_200_PLUS = 45135,
    BTN_1 = 12495,
    BTN_2 = 6375,
    BTN_3 = 31365,
    BTN_4 = 4335,
    BTN_5 = 14535,
    BTN_6 = 23205,
    BTN_7 = 17085,
    BTN_8 = 19125,
    BTN_9 = 21165,
    LONG_PRESS = 65535
} IR_REMOTE_KEYS;

static int FINGER_PINS[FINGER_COUNT] = {2, 3, 4, 5, 6};

static int FINGER_MAX_DEGS[FINGER_COUNT] = {90, 100, 100, 100, 100};

static Servo finger_servos[FINGER_COUNT];
static int current_finger_states[FINGER_COUNT] = {0};
//prevent from receiving constantly coming signal of rock, causing action not finished
static bool thumb_collision_lock = false;
static bool serial_port_data_lock = false;
IRrecv irrecv(IR_RECV_PIN); // Initiate IR signal input
unsigned long idle_ts;

bool check_thumb_collision(int new_finger_states[]) {
    return new_finger_states[THUMB] > CAUSE_COLLISION_DEG && (new_finger_states[INDEX] > CAUSE_COLLISION_DEG);
}

bool is_equal_to_prev_finger_states(int new_finger_states[]) {
    for (int i = 0; i < FINGER_COUNT; i++) {
        if (new_finger_states[i] != current_finger_states[i]) return false;
    }
    return true;
}

// Single-finger movement
void move_finger(int finger, int percentage) {
    idle_ts = millis();
    float max = FINGER_MAX_DEGS[finger], min = 0;
    //index and pinky finger initial position at degree 180
    if(finger == INDEX || finger == PINKY) {
        max = 180 - FINGER_MAX_DEGS[finger];
        min = 180;
    }
    //check thumb collision
    if (finger == THUMB && percentage >= CAUSE_COLLISION_DEG && current_finger_states[INDEX] >= CAUSE_COLLISION_DEG) {
        percentage = THUMB_COLLISION_MAX_PERCENTAGE;
    }
    float degree = (max - min) * percentage / 100 + min;
    //enum Finger are int
    int rounded_deg = (int) (degree + 0.5);

    finger_servos[finger].write(rounded_deg);
    //prevent servo malfunction when fingers move together
    delay(SERVO_MALFUNCTION_PREVENTION_DELAY);
    //keep track of current state
    current_finger_states[finger] = percentage;
}

// multi-finger control
void move_fingers(int fingers_states[]) {
    for (int i = 0; i < FINGER_COUNT; i++) {
      move_finger((Finger)i, fingers_states[i]);
    }
}

void move_fingers_with_collision_checking(int finger_states[]) {
    if (check_thumb_collision(finger_states)) {
        //save thumb state
        int thumb_finger_state = finger_states[THUMB];
        //TODO: Collision max degree for thumb and define above
        thumb_collision_lock = true;
        finger_states[THUMB] = 0;
        move_fingers(finger_states);
        delay(THUMB_COLLISION_WAIT);
        //move thumb using saved thumb state
        move_finger(THUMB, THUMB_COLLISION_MAX_PERCENTAGE);
        thumb_collision_lock = false;
    } else {
        move_fingers(finger_states);
    }
}

void move_fingers_with_ir_cmd(word remote_signal_code) {
    switch(remote_signal_code) {
        case BTN_0: {
            Serial.println("BTN_0 received: making paper gesture");
            reset_finger_states();
            break;
        }
        case BTN_1: {
            Serial.println("BTN_1 received: moving thumb");
            if (current_finger_states[THUMB]) move_finger(THUMB, 0);
            else move_finger(THUMB, 100); 
            break;
        }
        case BTN_2: {
            Serial.println("BTN_2 received: moving index finger");
            move_finger(INDEX, 100 - current_finger_states[INDEX]);
            break;
        }
        case BTN_3: {
            Serial.println("BTN_3 received: moving middle finger");
            move_finger(MIDDLE, 100 - current_finger_states[MIDDLE]);
            break;
        }
        case BTN_4: {
            Serial.println("BTN_4 received: moving ring finger");
            move_finger(RING, 100 - current_finger_states[RING]);
            break;
        }
        case BTN_5: {
            Serial.println("BTN_5 received: moving pinky");
            move_finger(PINKY, 100 - current_finger_states[PINKY]);
            break;
        }
        case BTN_6: {
            Serial.println("BTN_6 received: making finger wave");
            reset_finger_states();
            delay(MOVEMENT_WAITING_TIME * 2);
            start_finger_wave();
            break;
        }
        case BTN_7: {
            Serial.println("BTN_7 received: making scissor gesture");
            gesture_scissor();
            break;
        }
        case BTN_8: {
            Serial.println("BTN_8 received: Love gesture");
            int finger_states_love[FINGER_COUNT] = {0, 0, 100, 100, 0};
            move_fingers(finger_states_love);
            break;
        }
        case BTN_9: {
            Serial.println("BTN_9 received: doing rock gesture");
            reset_finger_states();
            delay(MOVEMENT_WAITING_TIME * 2);
            gesture_rock();
            break;
        }
        case LONG_PRESS: {
            Serial.println("LONG_PRESS received: doing nothing");
            break;
        }
        default: // Error
            Serial.print("Unhandled IR controller command received:" );
            Serial.println(remote_signal_code);
            break;
    }
}

void gesture_rock(){
    int finger_states_rock_stage[FINGER_COUNT] = {100, 100, 100, 100, 100};
    if (!is_equal_to_prev_finger_states(finger_states_rock_stage)) {
        move_fingers_with_collision_checking(finger_states_rock_stage);
    }
}

void gesture_scissor(){
    int finger_states_scissor[FINGER_COUNT] = {100, 0, 0, 100, 100};
    move_fingers(finger_states_scissor);
}

void reset_finger_states(){
    int relaxed_finger_states[FINGER_COUNT] = {0};
    move_fingers(relaxed_finger_states);
}


void handle_serial_cmd() {
    if (Serial.available() > 0 && !thumb_collision_lock && !serial_port_data_lock) {
        int count = 0;
        serial_port_data_lock = true;
        unsigned char serial_data = Serial.read();
        unsigned char a = 0b10000000;
        int finger_states[FINGER_COUNT];
        Serial.println("gesture: ");        
        for (int i = 0; i < FINGER_COUNT; i ++){
            finger_states[i] = ((serial_data & a) >> (7 - i)) * 100;
            Serial.println(finger_states[i]);
            a >>= 1;
        }


        if (!is_equal_to_prev_finger_states(finger_states)) {
            move_fingers(finger_states);
            delay(MOVEMENT_WAITING_TIME);
        }

        serial_port_data_lock = false;
    }
}


void handle_ir_remote_cmd() {
    decode_results remote_signal; // Save signal structure
    word remote_signal_code;  //remote_signal_code for IR remote control

    if (irrecv.decode(&remote_signal)) {
        remote_signal_code = remote_signal.value, HEX;
        move_fingers_with_ir_cmd(remote_signal_code);
        irrecv.resume();
    }
}

void start_finger_wave() {
    int finger_states_wave[FINGER_COUNT] = {0};
    for (int i = 0; i < FINGER_COUNT; i++) {
        move_finger((Finger)i, 100);
        delay(MOVEMENT_WAITING_TIME);
    }
    for (int i = FINGER_COUNT - 1; i >= 0; i--) {
        move_finger((Finger)i, 0);
        delay(MOVEMENT_WAITING_TIME);
    }
}

void setup() {
    Serial.println("Setup hand controller");
    // Record servos for the fingers
    for (int i = 0; i < FINGER_COUNT; i++) {
        finger_servos[i].attach(FINGER_PINS[i]);
    }
    reset_finger_states();

    #ifdef ENABLE_SERIAL_PORT_CONTROL
    Serial.begin(SERIAL_PORT);  // Connect to serial port, baud rate is 9600
    #endif

    #ifdef ENABLE_IR_REMOTE_CONTROL
    irrecv.blink13(true);       // If signal is received, then pin13 LED light blink
    irrecv.enableIRIn();        // Enable the IR receiver pin
    #endif
    Serial.println("Setup completed" ) ;
}

void loop() {
    // Reset all fingers if no command is received after IDLE_TIMEOUT
    if (millis() - idle_ts > IDLE_TIMEOUT) {
        idle_ts = millis();
        reset_finger_states();
        Serial.println("Bending the fingers too long, Release!");
    }
    //control by serial port
    #ifdef ENABLE_SERIAL_PORT_CONTROL
    handle_serial_cmd();
    #endif
    //control by IR remote
    #ifdef ENABLE_IR_REMOTE_CONTROL
    handle_ir_remote_cmd();
    #endif
}
