# If press 'i', using image processing method
# if press 'c', using pre-trained CNN model

# import packages
import numpy as np
import cv2
from utility.gesture_detection import *
from utility.frame_read import read_frame
import mxnet as mx
import warnings
import pickle
from utility.neural_network import *


# # Arudino
# import serial
# #open arudino ide and check the serial port num (appears on the lower right corner)
# serial_port_num ='1460'
# ser = serial.Serial('/dev/cu.usbserial-' + serial_port_num, 9600)

# gesture
paper = 0b00000000
rock = 0b11111000
scissor = 0b10011000

# get the reference to the camera
camera = cv2.VideoCapture(0)

# initialize start parameter, if 'start_recording = True'. start reading camera and prediction
start_image_processing = False
start_CNN = False
collecting_data = False
training_model = False
CNN_prediction =False

img_num = 0
training_data = {'scissor':[], 'rock':[], 'paper':[]}
img_num_base = 10

if __name__ == '__main__':

    while(True):
        # get region of interest "roi"
        roi = read_frame(camera)

        if start_image_processing:
            #  index number equals to the number of acute angle for gesture 
            gestures = ['rock', 'scissor', 'scissor' , 'unknown', 'paper', 'paper']
            # detect gesture based on skin
            thresh = detect_bodyskin(roi)  # TODO: rename
            # show segmented gesture
            cv2.imshow("gesture_threshold", thresh)
            # get a list of contours for gesture
            contours = get_contours(thresh.copy())
            if len(contours) == 0:
                pass
            # find the contour which have largest area in contour list
            else:
                largecont = max(contours, key = lambda contour: cv2.contourArea(contour)) 

                # get a convex contour of hand
                hull = cv2.convexHull(largecont, returnPoints = False)
                defects = cv2.convexityDefects(largecont, hull)
                # get convex contour and return gesture name
                # if verbose = Ture, auxiliary lines and auxiliary points using for prediction will show; otherwise, not
                if defects is not None:
                    img, ndefects = get_defects_count(roi, largecont, defects, verbose = True)
                    print(gestures[ndefects]) if ndefects < len(gestures) else 'unkonwn' 

        elif start_CNN:
            # a list of all classes of gestures
            label = ['scissor', 'rock', 'paper']
            text = ["Press any key to start collecting SCISSOR gesture.", "Press any key to start collecting ROCK gesture.", "Press any key to start collecting PAPER gesture."]
            # preprocess image of gesture
            img = detect_bodyskin(roi)
            # show preprossed image
            cv2.imshow('img_mask', img)
            # get a list of contours for gesture
            contours = get_contours(img.copy())
            if len(contours) == 0:
                pass
            # find the contour which have largest area in contour list
            else:
                largecont = max(contours, key = lambda contour: cv2.contourArea(contour))
            # down sample image
            img = down_sample(img, largecont)
            cv2.imshow('down_sample_img', img)
            if collecting_data:
                if img_num%img_num_base == 0:
                    input(text[int(img_num/100)])
                elif img_num//img_num_base == 0:
                    training_data[label[0]].append(img)
                    print('Collecting SCISSOR data: {}'.format(img_num))
                elif img_num//img_num_base == 1:
                    training_data[label[1]].append(img)
                    print('Collecting ROCK data: {}'.format(img_num-img_num_base))
                elif img_num//img_num_base == 2:
                    training_data[label[2]].append(img)
                    print('Collecting PAPER data: {}'.format(img_num-img_num_base*2))
            img_num += 1

            if training_model:
                input("Press any key to start model trianing.")
                train_data = generate_data(training_data, label)
                # CNN network
                net, accuracy = train_model(train_data) 
                print(accuracy) 
                training_model = False
                CNN_prediction = True

            if CNN_prediction: 
                # put preprocessed image into CNN and get probability array in shape (1, number of class)
                outputs = mx.nd.softmax(net(mx.nd.array(img.reshape(1,1,128,128)))).asnumpy()
                # label index 
                gesture = np.argmax(outputs[0])
                # print the class 
                print(label[gesture])
                # ser.write(bytes(label[gesture]))

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF
        # before typing the key, make sure stay in video window
        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break
        # press "s" on keyboard, start prediction
        if keypress == ord("s"):   # TODO: no s 
            method_choose = input('Press i to start image processing method, press c to start CNN method: ')
            if method_choose == 'i':
                start_image_processing = True
                start_CNN = False
            elif method_choose == 'c':
                start_image_processing = False
                start_CNN = True
                collecting_data = True
        if img_num == 3*img_num_base:
            # with open('./trainingdata.pickle', 'wb') as handle:
            #     pickle.dump(training_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            # print('Successfully Saving Data')
            collecting_data = False
            training_model = True
                    
# camera off
camera.release()
