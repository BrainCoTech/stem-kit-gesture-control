import mxnet as mx
import numpy as np
import cv2
import math


class gestureDetect:

    __gestures = ['roc', 'sic', 'sic' , 3, 'pap', 'pap']

    def __init__(self):
        self.__color = (255,0,0)
        self.__bgKernel = (3, 3)
        self.__blurKernel = (5, 5)
        self.__angleThreshold = math.pi/2

    def _remove_background(self, frame):
        fgbg = cv2.createBackgroundSubtractorKNN()
        fgmask = fgbg.apply(frame)
        kernel = np.ones(self.__bgKernel, np.uint8)
        fgmask = cv2.erode(fgmask, kernel, iterations=1)
        res = cv2.bitwise_and(frame, frame, mask=fgmask)
        return res

    def _bodyskin_detetc(self, frame):
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        (_, cr, _) = cv2.split(ycrcb)
        cr1 = cv2.GaussianBlur(cr, self.__blurKernel, 0) 
        _, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 
        return skin

    def _get_contours(self,array):
        contours, _ = cv2.findContours(array, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        return contours

    def _get_defects_count(self, array, contour, defects, verbose):
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
            if angle <= self.__angleThreshold :
                ndefects = ndefects + 1
                if verbose:
                    cv2.circle(array, tuple(far), 3, self.__color, -1)
            if verbose:
                cv2.line(array, tuple(beg), tuple(end), self.__color, 1)
        return array, ndefects

    # def grdetect(self, array, verbose = False): 
    #     copy = array.copy()
    #     array_n = self._remove_background(array)
    #     thresh = self._bodyskin_detetc(array)

    #     # contours is a list of all the contours in the image
    #     contours = self._get_contours(thresh.copy())
    #     largecont = max(contours, key = lambda contour: cv2.contourArea(contour))

    #     hull = cv2.convexHull(largecont, returnPoints = False)
    #     defects = cv2.convexityDefects(largecont, hull)

    #     if defects is not None:
    #         copy, ndefects = self._get_defects_count(copy, largecont, defects, verbose = verbose)
    #         return self.__gestures[ndefects] if ndefects < len(self.__gestures) else 'nothing'

    def grdetect(self, array, verbose = True): 
        copy = array.copy()
        array_nbg = self._remove_background(copy)
        thresh = self._bodyskin_detetc(array_nbg)

        # contours is a list of all the contours in the image
        contours = self._get_contours(thresh.copy())
        largecont = max(contours, key = lambda contour: cv2.contourArea(contour))

        hull = cv2.convexHull(largecont, returnPoints = False)
        defects = cv2.convexityDefects(largecont, hull)

        if defects is not None:
            array, ndefects = self._get_defects_count(array, largecont, defects, verbose = verbose)
            return self.__gestures[ndefects] if ndefects < len(self.__gestures) else 'nothing'  

