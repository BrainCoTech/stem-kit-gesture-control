i# If press 'i', using image processing method
# if press 'c', using pre-trained CNN model
import cv2
import serial
from enum import Enum
from gesture_detection_with_countours import detect_with_contours
from utility.gesture import Gesture
from utility.camera import read_frame
from gesture_detection_with_cnn import predict_with_cnn

_SERIAL_PORT_NUMBER = '14230'
_USE_ARDUINO = False
# gesture
_GESTURE_TO_SERIAL_COMMANDS = {Gesture.paper: 0b10011000, Gesture.rock: 0b00000000, Gesture.scissor: 0b11111000}

should_quit = False

class Modes(Enum):
    unstarted = 0
    contours = 1
    cnn = 2


mode = Modes.unstarted

def handle_key_press():
    global should_quit, mode
    
    keypress = cv2.waitKey(1) & 0xFF
    if keypress == ord('q'):
        should_quit = True

    elif keypress == ord('i'):
        mode = Modes.contours
        
    elif keypress == ord('c'):
        mode = Modes.cnn
        
        
def write_gesture_to_serial_port(gesture):
    if gesture is not Gesture.unknown:
        ser = serial.Serial('/dev/cu.usbserial-' + _SERIAL_PORT_NUMBER, 9600)
        ser.write(bytes([_GESTURE_TO_SERIAL_COMMANDS[gesture]]))

if __name__ == "__main__":

    camera = cv2.VideoCapture(0)
    cv2.namedWindow("Video", 0)

    count = 0

    while not should_quit:
        roi = read_frame(camera)
        handle_key_press()
        gesture_detected = Gesture.unknown

        if mode is Modes.contours:
            gesture_detected, img = detect_with_contours(roi)
            count += 1

        elif mode is Modes.cnn:
            gesture_detected, img = predict_with_cnn(roi)
            count += 1

        else:  # not started
            img = roi
                
        if _USE_ARDUINO and count == 10:
            write_gesture_to_serial_port(gesture_detected)
            print("Gesture detected with " + mode.name +": " + gesture_detected.name)
            count = 0
            
        cv2.imshow("Video", img)

    camera.release()
    cv2.destroyAllWindows()
