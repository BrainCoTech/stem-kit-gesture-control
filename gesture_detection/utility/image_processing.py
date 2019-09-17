import numpy as np
import cv2
import math

# parameters 
_bgKernel = (3, 3)
_blurKernel = (5, 5)
_erodeKernel = (7, 7)
_angleThreshold = math.pi/2


# segment hand shape based on skin detection and return a binary image 
def detect_body_skin(frame):
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    (_, cr, _) = cv2.split(ycrcb)
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


def down_sample(img, large_contour):  # TODO: Pass down sample target size as a parameter
    # region for hand
    img_height = img.shape[0]
    x = large_contour[:, 0, 0]
    x_min = min(x)
    x_max = max(x)
    y = large_contour[:, 0, 1]
    y_min = min(y)
    y_max = max(y)
    diff = (x_max-x_min) - (y_max-y_min)
    if diff > 0:
        img = img[x_min:x_max, int(img_height/2-(x_max-x_min)/2):int(img_height/2+(x_max-x_min)/2)]
    else:
        img = img[int(img_height/2-(y_max-y_min)/2):int(img_height/2+(y_max-y_min)/2), y_min:y_max]
    img = cv2.resize(img, (128, 128))
    return img


# get the number of acute angle
def get_defects_count(img, contour, defects, draw_on_figure=True):
    n_defects = 0
    for i in range(defects.shape[0]):
        s, e, f, _ = defects[i, 0]
        beg = contour[s][0]
        end = contour[e][0]
        far = contour[f][0]
        a = np.linalg.norm(beg - end)
        b = np.linalg.norm(beg - far)
        c = np.linalg.norm(end - far)
        angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
        if angle <= _angleThreshold:
            n_defects = n_defects + 1
            if draw_on_figure:
                cv2.circle(img, tuple(far), 3, (255, 0, 0), -1)
                cv2.circle(img, tuple(beg), 3, (0, 255, 0), -1)
                cv2.circle(img, tuple(end), 3, (125, 125, 125), -1)
        if draw_on_figure:
            cv2.line(img, tuple(beg), tuple(end), (255, 0, 0), 1)

    return img, n_defects


def preprocess_for_cnn(roi):
    # preprocess image of gesture
    img = detect_body_skin(roi)
    # get a list of contours for gesture
    contours = get_contours(img.copy())
    if len(contours) != 0:
        large_contour = max(contours, key=lambda contour: cv2.contourArea(contour))
    # down sample image
    img = down_sample(img, large_contour)
 
    return img


def get_largest_contour(contours):
    return max(contours, key=lambda contour: cv2.contourArea(contour))


def concatenate_images(gray_scale_img, color_img):
    # concatenate a gray scale img and a 3 channels img
    gray_scale_img = cv2.cvtColor(gray_scale_img, cv2.COLOR_GRAY2BGR)
    images = np.concatenate((gray_scale_img, color_img), axis=1)
    return images


def get_convex_hull(largest_contour):
    return cv2.convexHull(largest_contour, returnPoints=False)
