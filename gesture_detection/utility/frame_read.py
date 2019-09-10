import cv2
import imutils

# parameters
_roi_size = 240

# read frame from camera, return back frame and a segmented region where to put hands in 
def read_frame(camera):
# def read_frame(camera):
    # get the current frame
    (grabbed, frame) = camera.read()  # shape (720,1280)
    # resize the frame
    frame = imutils.resize(frame, width=800)
    # flip the frame so that it is not the mirror view
    frame = cv2.flip(frame, 1)
    # frame size
    h = frame.shape[0]
    w = frame.shape[1]
    # get the region of interest (ROI)
    top = 20
    bottom = int(top + _roi_size)
    right = int(w/2 - _roi_size/2)
    left = int(right + _roi_size)
    # clone the frame
    clone = frame.copy()
    # get the region of interest (ROI)
    roi = frame[top:bottom, right:left]
    # draw a region to segment hand
    cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)
    # display the frame 
    cv2.imshow("Video Feed", clone)
    return roi
