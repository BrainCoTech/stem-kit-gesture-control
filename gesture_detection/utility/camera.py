import cv2
import imutils

# parameters
_ROI_SIZE = 240
_TOP, _LEFT = 20, 450

# read frame from camera, return back frame and a segmented region where to put hands in 
def read_frame(camera):
# def read_frame(camera):
    # get the current frame
    (grabbed, frame) = camera.read()  # shape (720,1280)
    # resize the frame
    frame = imutils.resize(frame, width=800)
    # flip the frame so that it is not the mirror view
    frame = cv2.flip(frame, 1)
    # clone the frame
    clone = frame.copy()
    # get the region of interest (ROI)
    roi = frame[_TOP:_TOP+_ROI_SIZE, _LEFT:_LEFT+_ROI_SIZE]
    # draw a region to segment hand
    cv2.rectangle(clone, (_LEFT, _TOP), (_LEFT+_ROI_SIZE, _TOP+_ROI_SIZE), (0,255,0), 2)
    # display the frame 
    cv2.imshow("Video Feed", clone)
    return roi