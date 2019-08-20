import numpy as np
import cv2
import math

# parameters can be adjusted
_color = (255,0,0)
_bgKernel = (3, 3)
_blurKernel = (5, 5)
_erodeKernel = (10, 10)
_angleThreshold = math.pi/2

# segment hand shape based on skin detection and return a binary image 
def detect_bodyskin(frame):
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    (_, cr, _) = cv2.split(ycrcb)
    cr1 = cv2.GaussianBlur(cr, _blurKernel, 0) 
    _, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((10,10), np.uint8)
    skin = cv2.erode(skin, kernel, iterations=1)
    skin = cv2.dilate(skin, kernel, iterations=1)
    return skin
    
# get a list of contours of binary img
def get_contours(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours

# calculate center coordinate of contour
def get_center(largecont):
    M = cv2.moments(largecont)
    center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
    return center

# get the number of acute angle
def get_defects_count(img, contour, defects, verbose):
    ndefects = 0
    for i in range(defects.shape[0]):
        s,e,f,_ = defects[i,0]
        beg = contour[s][0]
        end = contour[e][0]
        far = contour[f][0]
        a = np.linalg.norm(beg - end)
        b = np.linalg.norm(beg - far)
        c = np.linalg.norm(end - far)
        angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
        if angle <= _angleThreshold :
            ndefects = ndefects + 1
            if verbose:
                cv2.circle(img, tuple(far), 3, _color, -1)
                cv2.circle(img, tuple(beg), 3, (0,255,0), -1)
                cv2.circle(img, tuple(end), 3, (125,125,125), -1)
        if verbose:
            cv2.line(img, tuple(beg), tuple(end), _color, 1) 
            cv2.imshow('img', img)     
    return img, ndefects


