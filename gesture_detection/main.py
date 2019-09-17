# If press 'i', using image processing method
# if press 'c', using pre-trained CNN model

# import packages
import cv2
import serial
from gesture_detection_with_countours import detect_with_coutours
from utility.camera import read_frame
from enum import Enum
from gesture_detection_with_cnn import predict_with_cnn

_SERIAL_PORT_NUMBER ='14240'
_USE_ARDUINO = False


class Gesture(Enum):
    unknown = -1
    paper = 0
    rock = 1
    scissor = 2

# gesture
_GESTURE_SERIAL_COMMANDS = {Gesture.paper: 0b00000000, Gesture.rock: 0b11111000, Gesture.scissor: 0b10011000}
       
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
        ser.write(_GESTURE_SERIAL_COMMANDS[gesture])


if __name__ == "__main__":

    camera = cv2.VideoCapture(0)
    cv2.namedWindow("Video",0)

    while(not should_quit):
        
        roi = read_frame(camera)

        handle_key_press()
        
        gesture_detected = Gesture.unknown
        
        if mode is Modes.contours:
            index, img = detect_with_coutours(roi)
            gesture_detected = Gesture(index)
            print("Gesture detected with contours:%s" % gesture_detected)

        elif mode is Modes.cnn:
            index, img = predict_with_cnn(roi)
            gesture_detected = Gesture(index)
            print("Gesture detected with CNN:%s" % gesture_detected)

        else: # unstarted
            img = roi
                
        if _USE_ARDUINO:
            write_gesture_to_serial_port(gesture_detected)
            
        cv2.imshow("Video", img)

    camera.release()
    cv2.destroyAllWindows()

