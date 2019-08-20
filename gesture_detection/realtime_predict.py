# If press 'i', using image processing method
# if press 'c', using pre-trained CNN model

# import packages
import numpy as np
import cv2
from utility.gesture_detection import *
from utility.frame_read import read_frame
import mxnet as mx
import warnings

# get the reference to the camera
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_EXPOSURE, 0.25)

# region of interest (ROI) coordinates, put hands in this region to record or read gesture
top, right, bottom, left = 10, 350, 225, 590

# initialize start parameter, if 'start_recording = True'. start reading camera and prediction
start_image_processing = False
start_CNN = False

# import CNN model 
symbol_file = './gesture_model-symbol.json'
params_file = './gesture_model-0000.params'

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    net = mx.gluon.SymbolBlock.imports(symbol_file, ['data'], params_file)

if __name__ == '__main__':
    while(True):
        # get video frame "clone"
        # get region of interest "roi"
        clone, roi = read_frame(camera, top, right, bottom, left)

        if start_image_processing:
            #  index number equals to the number of acute angle for gesture 
            _gestures = ['rock', 'scissor', 'scissor' , 'unknown', 'paper', 'paper']

            copy = roi.copy()
            # detect gesture based on skin
            thresh = detect_bodyskin(copy)
            # show segmented gesture
            cv2.imshow("gesture_threshold", thresh)
            # get a list of contours for gesture
            contours = get_contours(thresh.copy())
            # find the contour which have largest area in contour list
            largecont = max(contours, key = lambda contour: cv2.contourArea(contour)) 
            # get a convex contour of hand
            hull = cv2.convexHull(largecont, returnPoints = False)
            # return the number of acute angle
            defects = cv2.convexityDefects(largecont, hull)
            # return gesture name
            # if verbose = Ture, auxiliary lines and auxiliary points using for prediction will show; otherwise, not
            if defects is not None:
                img, ndefects = get_defects_count(roi, largecont, defects, verbose = True)
                print(_gestures[ndefects]) if ndefects < len(_gestures) else 'nothing' 

        elif start_CNN:
            # a list of all classes of gestures
            label = ['scissor', 'rock', 'paper']
            # preprocess image of gesture
            img_mask = detect_bodyskin(roi)
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
            method_choose = input('Press i to start image processing method, press c to start CNN method: ')
            if method_choose == 'i':
                start_image_processing = True
                start_CNN = False
            elif method_choose == 'c':
                start_image_processing = False
                start_CNN = True


# camera off
camera.release()
