import mxnet as mx
import numpy as np
import cv2
import math


#  index number equals to the number of sharp angle for gesture 
_gestures = ['rock', 'scissor', 'scissor' , 'unknown', 'paper', 'paper']
_color = (255,0,0)
_bgKernel = (3, 3)
_blurKernel = (5, 5)
_angleThreshold = math.pi/2

class GestureDetect:

    @staticmethod
    def __remove_background(frame):
        fgbg = cv2.createBackgroundSubtractorKNN()
        fgmask = fgbg.apply(frame)
        kernel = np.ones(_bgKernel, np.uint8)
        fgmask = cv2.erode(fgmask, kernel, iterations=1)
        res = cv2.bitwise_and(frame, frame, mask=fgmask)
        return res

    @staticmethod
    def __detect_bodyskin(frame):
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        (_, cr, _) = cv2.split(ycrcb)
        cr1 = cv2.GaussianBlur(cr, _blurKernel, 0) 
        _, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 
        return skin

    @staticmethod
    def __get_contours(img):
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        return contours

    @staticmethod
    def __get_defects_count(img, contour, defects, verbose):
        ndefects = 0
        for i in range(defects.shape[0]):
            s,e,f,_ = defects[i,0]
            beg = contour[s][0]
            end = contour[e][0]
            far = contour[f][0]
            # get eucledian distance
            a = np.linalg.norm(beg - end)
            b = np.linalg.norm(beg - far)
            c = np.linalg.norm(end - far)
            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) # * 57
            if angle <= _angleThreshold :
                ndefects = ndefects + 1
                if verbose:
                    cv2.circle(img, tuple(far), 3, _color, -1)
            if verbose:
                cv2.line(img, tuple(beg), tuple(end), _color, 1)
        return img, ndefects

    # def grdetect(self, img, verbose = False): 
    #     copy = img.copy()
    #     img = _remove_background(img)
    #     thresh = _detect_bodyskin(img)

    #     # contours is a list of all the contours in the image
    #     contours = _get_contours(thresh.copy())
    #     largecont = max(contours, key = lambda contour: cv2.contourArea(contour))

    #     hull = cv2.convexHull(largecont, returnPoints = False)
    #     defects = cv2.convexityDefects(largecont, hull)

    #     if defects is not None:
    #         copy, ndefects = _get_defects_count(copy, largecont, defects, verbose = verbose)
    #         return _gestures[ndefects] if ndefects < len(_gestures) else 'nothing'

    @staticmethod
    def grdetect(img, verbose = True): 
        copy = img.copy()
        img_nbg = GestureDetect.__remove_background(copy)
        thresh = GestureDetect.__detect_bodyskin(img_nbg)

        # contours is a list of all the contours in the image
        contours = GestureDetect.__get_contours(thresh.copy())
        largecont = max(contours, key = lambda contour: cv2.contourArea(contour))

        hull = cv2.convexHull(largecont, returnPoints = False)
        defects = cv2.convexityDefects(largecont, hull)

        if defects is not None:
            img, ndefects = GestureDetect.__get_defects_count(img, largecont, defects, verbose = verbose)
            return _gestures[ndefects] if ndefects < len(_gestures) else 'nothing'  

