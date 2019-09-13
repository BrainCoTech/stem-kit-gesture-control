# If press 'i', using image processing method
# if press 'c', using pre-trained CNN model

# import packages
import cv2
import serial
from image_processing_method import image_processing_func
from utility.frame_read import read_frame


_USE_ARDUINO = False

# gesture
PAPER = 0b00000000
ROCK = 0b11111000
SCISSOR = 0b10011000
gesture_names = [PAPER, ROCK, SCISSOR]
       
should_quit = False

from enum import Enum
class Modes(Enum):
    unstarted = 0
    contours = 1
    cnn = 2

mode = Modes.unstarted
             
def select_mode():
    
    keypress = cv2.waitKey(1) & 0xFF
    if keypress == ord('q'):
        should_quit = True

    elif keypress == ord('i'):
        mode = Modes.contours
        
    elif keypress == ord('c'):
        mode = Modes.cnn
        
def send_to_hand(res):
    serial_port_num ='14240'
    ser = serial.Serial('/dev/cu.usbserial-' + serial_port_num, 9600)
    ser.write(res)

    

if __name__ == "__main__":

    camera = cv2.VideoCapture(0)
    cv2.namedWindow("Video",0)

    while(not should_quit):

        roi = read_frame(camera)
        select_mode()
            
        if mode is Modes.contours:
            index, img = image_processing_func(roi)
            res =  gesture_names[index]
            
        elif mode is Modes.cnn:
            print("cnn")
        else: # unstarted
            img = read_frame(camera)

        if _USE_ARDUINO:
            send_to_hand(res)    
        
        cv2.imshow("Video", img)
        
    camera.release()
    cv2.destroyAllWindows()

