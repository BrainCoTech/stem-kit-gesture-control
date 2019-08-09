import numpy as np
import warnings
import cv2
import imutils
import math
from utility.gesture_detection import GestureDetect

# get the reference to the webcam
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_EXPOSURE, 0.25)

# region of interest (ROI) coordinates
top, right, bottom, left = 10, 350, 225, 590

# initialize num of frames

start_recording = False

if __name__ == '__main__':
    while(True):
        # get the current frame
        (grabbed, frame) = camera.read()
        # resize the frame
        frame = imutils.resize(frame, width = 700)
        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)
        # clone the frame
        clone = frame.copy()
        # get the ROI
        roi = frame[top:bottom, right:left]

        if start_recording:
            cur_gesture = GestureDetect.grdetect(roi)
            print(cur_gesture)

        # draw the segmented hand
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)
        # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)
        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF
        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break
                
        if keypress == ord("s"):
            start_recording = True

camera.release()
