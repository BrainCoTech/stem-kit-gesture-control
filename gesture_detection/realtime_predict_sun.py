# If press 'i', using image processing method
# if press 'c', using pre-trained CNN model

# import packages
import cv2
from image_processing_method import image_processing_method_result
from cnn import cnn_method
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

img_num = 0

if __name__ == '__main__':
    
    
    while(True):
    
        keypress = cv2.waitKey(1) & 0xFF

        if keypress == ord("q"):
            break
        
        if keypress == ord("i"):
            cur_gesture_index = image_processing_method()
            res = gesture_names[cur_gesture_index]
                
        elif keypress == ord("c"):
            collecting_data = input('Collecting data or not, input True or False:')    
            if collecting_data == 'True':
                data = collect_data(img)
            img_num += 1
            cur_gesture_index = cnn_method(camera, collecting_data, img_num)
            res = gesture_names[cur_gesture_index]
            
        if(_USE_ARDUINO):  

            ser.write(res)


    

                    
# camera off
camera.release()
