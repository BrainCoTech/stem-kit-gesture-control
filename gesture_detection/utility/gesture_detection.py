import numpy as np
import cv2
import math

# parameters 
_bgKernel = (3, 3)
_blurKernel = (5, 5)
_erodeKernel = (7, 7)
_angleThreshold = math.pi/2

def remove_background(frame):
    fgbg = cv2.createBackgroundSubtractorMOG2(history=20)
    fgmask = fgbg.apply(frame)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res

# gamma transfer
def gamma_trans(img,gamma):
    gamma_table=[np.power(x/255.0,gamma)*255.0 for x in range(256)]
    gamma_table=np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img,gamma_table)

# segment hand shape based on skin detection and return a binary image 
def detect_bodyskin(frame):
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    (_, cr, _) = cv2.split(ycrcb)
    cr1 = cv2.GaussianBlur(cr, _blurKernel, 0) 
    _, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones(_erodeKernel, np.uint8)
    skin = cv2.erode(skin, kernel, iterations=1)
    skin = cv2.dilate(skin, kernel, iterations=1)
    return skin

def adjusted_skin_detection(frame):
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    B = frame[:,:,0]
    G = frame[:,:,1]
    R = frame[:,:,2]
    Y = 0.299*R + 0.587*G + 0.114*B
    Y_value = np.mean(Y)
    print(Y_value)
    if Y_value<200:
        cr0 = (R-Y)*0.713 + 128
    else:
        cr0 = np.multiply(((R-Y)**2*0.713),((-5000/91)*(Y-200)**(-2)+7)) + 128
    cr = np.array(cr0, dtype=np.uint8)
    cr1 = cv2.GaussianBlur(cr, _blurKernel, 0) 
    _, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones(_erodeKernel, np.uint8)
    skin = cv2.erode(skin, kernel, iterations=1)
    skin = cv2.dilate(skin, kernel, iterations=1)
    return skin

    
# get a list of contours of binary img
def get_contours(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours

def down_sample(img, largecont):
    # region for hand
    l = img.shape[0]
    x = largecont[:,0,0]
    x_min = min(x)
    x_max = max(x)
    y = largecont[:,0,1]
    y_min = min(y)
    y_max = max(y)
    diff = (x_max-x_min) - (y_max-y_min)
    if diff>0:
        img = img[x_min:x_max, int(l/2-(x_max-x_min)/2):int(l/2+(x_max-x_min)/2)]
    else:
        img = img[int(l/2-(y_max-y_min)/2):int(l/2+(y_max-y_min)/2), y_min:y_max]
    img = cv2.resize(img, (128,128))
    return img

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
                cv2.circle(img, tuple(far), 3, (255,0,0), -1)
                cv2.circle(img, tuple(beg), 3, (0,255,0), -1)
                cv2.circle(img, tuple(end), 3, (125,125,125), -1)
        if verbose:
            cv2.line(img, tuple(beg), tuple(end), (255,0,0), 1) 

    return img, ndefects

def preprocess_cnn_img(roi):
    # preprocess image of gesture
    img = detect_bodyskin(roi)   
    # get a list of contours for gesture
    contours = get_contours(img.copy())
    if len(contours) != 0:
        largecont = max(contours, key = lambda contour: cv2.contourArea(contour))
    # down sample image
    img = down_sample(img, largecont)
 
    return img

