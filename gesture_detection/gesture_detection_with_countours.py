#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 13:47:49 2019

@author: jiaxinsun
"""

import cv2
import numpy as np
from utility.image_processing import detect_bodyskin, get_contours, get_defects_count


# gesture dictionary
# ndefects: [gesture num, gesture name]
UNKOWN = -1
gestures = {0:[1, "rock"], 1:[2, "scissor"], 2:[2, "scissor"], 3:[UNKOWN, "unkown"], 4:[0, "paper"], 5:[0, "paper"]}


def detect_with_coutours(roi):
    
    # detect gesture based on skin
    thresh_img = detect_bodyskin(roi)
    imgs = concatenate_imgs(thresh_img, roi)

    # get a list of contours for gesture
    contours = get_contours(thresh_img.copy())
    
    if len(contours) > 0:
    
        # find the contour which have largest area in contour list
        largest_contour = max(contours, key = lambda contour: cv2.contourArea(contour)) 
        
        # get a convex contour of hand
        hull = cv2.convexHull(largest_contour, returnPoints = False)
        defects = cv2.convexityDefects(largest_contour, hull)
        
        # get convex contour and return gesture name
        # if verbose = Ture, auxiliary lines and auxiliary points using for prediction will show; otherwise, not
        if defects is not None:
            defects_img, ndefects = get_defects_count(roi, largest_contour, defects)               
            #print(gestures[ndefects])   
            imgs = concatenate_imgs(thresh_img, defects_img)
            
            # if the number of defects greater than five, set ndefects = UNKOWN
            if ndefects > 5:
                ndefects = UNKOWN
                
            #print(gestures[ndefects][1])    
            return gestures[ndefects][0], imgs   
        
    return gestures[UNKOWN][0], imgs



def concatenate_imgs(gray_scale_img, color_img):
    #concatenate a gray scale img and a 3 channels img
    gray_scale_img = cv2.cvtColor(gray_scale_img, cv2.COLOR_GRAY2BGR)
    imgs = np.concatenate((gray_scale_img, color_img), axis=1)
    
    return imgs

        


    
    
    