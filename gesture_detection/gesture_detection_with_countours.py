#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 13:47:49 2019

@author: jiaxinsun
"""

import cv2
from utility.image_processing import detect_bodyskin, get_contours, get_defects_count, get_largest_contour, get_convex_hull, concatenate_imgs
from Gesture.py import Gesture


Gestures = [Gesture.rock, Gesture.scissor, Gesture.scissor, Gesture.unkown, Gesture.paper]


def detect_with_coutours(roi):
    
    # detect gesture based on skin
    thresh_img = detect_bodyskin(roi)

    # get a list of contours for gesture
    contours = get_contours(thresh_img.copy())
    
    if len(contours) > 0:
    
        # find the contour which have largest area in contour list
        largest_contour = get_largest_contour(contours) 
        
        # get a convex contour of hand
        hull = get_convex_hull(largest_contour)
        
        # get defects
        defects = cv2.convexityDefects(largest_contour, hull)


        # if defects exist, then get the number of defects 
        if defects is not None:
            
            defects_img, ndefects = get_defects_count(roi, largest_contour, defects)    
        
            imgs = concatenate_imgs(thresh_img, defects_img)
            
            # if the number of defects <=5, return
            if ndefects <= 5: 
                
                return Gestures[ndefects].val, imgs
            
        
    
    imgs = concatenate_imgs(thresh_img, roi)
    return Gesture.unkown.val, imgs




    
    
    