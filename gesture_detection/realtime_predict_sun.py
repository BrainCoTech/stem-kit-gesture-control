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
       
           
BREAK_LOOP = 0
IMAGE_PROCESSING = 1
CNN = 2

             
def choose_method(cur_method_choose):
    
    keypress = cv2.waitKey(1) & 0xFF
    if keypress == ord('q'):  
        method_choose = BREAK_LOOP

    elif keypress == ord('i'):
        method_choose = IMAGE_PROCESSING
        
    elif keypress == ord('c'):
        method_choose = CNN
        
    else:
        method_choose = cur_method_choose
    
    return method_choose
        

        
                    
def send_to_hand(res):
    serial_port_num ='14240'
    ser = serial.Serial('/dev/cu.usbserial-' + serial_port_num, 9600)
    ser.write(res)

    

if __name__ == "__main__":

    camera = cv2.VideoCapture(0)
    cv2.namedWindow("Video",0)
    method = None

    while(True):

        roi = read_frame(camera)
        method = choose_method(method)
        
        if method == None:
            img = read_frame(camera)

        elif method == IMAGE_PROCESSING:
            index, img = image_processing_func(roi)
            res =  gesture_names[index]
            
        elif method == CNN:
            print("cnn")
                        
        elif method == BREAK_LOOP:
            break

       
        if _USE_ARDUINO:
            send_to_hand(res)    
        
        cv2.imshow("Video", img)
     
    camera.release()
    cv2.destroyAllWindows()

