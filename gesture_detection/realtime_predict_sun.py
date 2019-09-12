# If press 'i', using image processing method
# if press 'c', using pre-trained CNN model

# import packages
import cv2
from image_processing_method import image_processing_method_result
import serial



_USE_ARDUINO = True
if _USE_ARDUINO:
    serial_port_num ='14240'
    ser = serial.Serial('/dev/cu.usbserial-' + serial_port_num, 9600)



# get the reference to the camera
camera = cv2.VideoCapture(0)


# gesture
paper = 0b00000000
rock = 0b11111000
scissor = 0b10011000
gesture_names = [paper, rock, scissor]



def main():
    
    
    while(True):
    
        keypress = cv2.waitKey(1) & 0xFF

        if keypress == ord("q"):
            break
        
        if keypress == ord("i"):
            
            res = gesture_names[image_processing_method_result()]
                
        elif keypress == ord("c"):
            
            res = gesture_names[cnn_method_result()]
            
        if(_USE_ARDUINO):  

            ser.write(res)


    

                    
# camera off
camera.release()
