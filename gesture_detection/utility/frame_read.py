import cv2
import imutils

# parameters
_roi_size = 240
_top, _left = 20, 700
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
    roi = frame[_top:_top+_roi_size, _left+_roi_size:_left]
    # draw a region to segment hand
    cv2.rectangle(clone, (_left, _top), (_left+_roi_size, _top+_roi_size), (0,255,0), 2)
    # display the frame 
    cv2.imshow("Video Feed", clone)
    return roi
