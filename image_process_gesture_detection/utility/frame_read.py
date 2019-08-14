import cv2
import imutils

# read frame from camera, return back frame and a segmented region where to put hands in 
def read_frame(camera, top, right, bottom, left):
    # get the current frame
    (grabbed, frame) = camera.read()
    # resize the frame
    frame = imutils.resize(frame, width = 700)
    # flip the frame so that it is not the mirror view
    frame = cv2.flip(frame, 1)
    # clone the frame
    clone = frame.copy()
    # get the height and width of the frame
    (height, width) = frame.shape[:2]
    # get the region of interest (ROI)
    roi = frame[top:bottom, right:left]
    # draw a region to segment hand
    cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)
    # display the frame 
    cv2.imshow("Video Feed", clone)
    return clone, roi