# preprocess images to extract features
# feed features in and train a CNN model to predict gestures

# import packages
import mxnet as mx
import numpy as np
import cv2
import warnings
import utility

# import CNN model 
symbol_file = './gesture_model-symbol.json'
params_file = './gesture_model-0000.params'

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    net = mx.gluon.SymbolBlock.imports(symbol_file, ['data'], params_file)

# get the reference to the webcam
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_EXPOSURE, 0.25)

# region of interest (ROI) coordinates
top, right, bottom, left = 10, 350, 225, 590

# initialize start parameter, if 'start_recording = True'. start reading camera and prediction
start_recording = False

# a list of all classes of gestures
label = ['scissor', 'rock', 'paper']

while(True):
    # get video frame "clone"
    # get region of interest "roi"
    clone, roi = utility.read_frame(camera, top, right, bottom, left)

    if start_recording:
        # preprocess image of gesture
        img_mask = utility.detect_bodyskin(roi)
        # show preprossed image
        cv2.imshow('img_mask', img_mask)
        # put preprocessed image into CNN and get probability array in shape (1, number of class)
        outputs = mx.nd.softmax(net(mx.nd.array(img_mask.reshape(1,1,215,240)))).asnumpy()
        # label index 
        gesture = np.argmax(outputs[0])
        # print the class 
        print(label[gesture])

    # observe the keypress by the user
    keypress = cv2.waitKey(1) & 0xFF
    # before typing the key, make sure stay in video window
    # if the user pressed "q", then stop looping
    if keypress == ord("q"):
        break
    # press "s" on keyboard, start prediction        
    if keypress == ord("s"):
        start_recording = True
# camera off
camera.release()
