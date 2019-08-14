# use only image processing method to predict gesture
# classify gestures based on the number of acute angle
# import packages
import numpy as np
import cv2
from utility.gesture_detection import GestureDetect
from utility.frame_read import read_frame

# get the reference to the camera
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_EXPOSURE, 0.25)

# region of interest (ROI) coordinates, put hands in this region to record or read gesture
top, right, bottom, left = 10, 350, 225, 590

# initialize start parameter, if 'start_recording = True'. start reading camera and prediction
start_recording = False

if __name__ == '__main__':
    while(True):
        # get video frame "clone"
        # get region of interest "roi"
        clone, roi = read_frame(camera, top, right, bottom, left)

        if start_recording:
            # if 'v'erbose = Ture', auxiliary lines and auxiliary points using for prediction will show; otherwise, not
            cur_gesture = GestureDetect.grdetect(roi, verbose=True)
            print(cur_gesture)

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
