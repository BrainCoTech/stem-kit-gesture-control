# If press 'i', using image processing method
# if press 'c', using pre-trained CNN model

# import packages
import cv2
import serial
from image_processing_method import image_processing_func
from utility.frame_read import read_frame
from enum import Enum
from cnn import cnn_method

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
     
def select_mode():
    global should_quit
    global mode
    
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
        select_mode()
        
        gesture_detected = Gesture.unknown
        
        if mode is Modes.contours:
            index, img = image_processing_func(roi)
            gesture_detected =  Gesture(index)
            
        elif mode is Modes.cnn:
            index = cnn_method(roi)
        else: # unstarted
            img = roi

                
        if _USE_ARDUINO:
            write_gesture_to_serial_port(gesture_detected)    
            
        cv2.imshow("Video", img)

    camera.release()
    cv2.destroyAllWindows()

