#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 13:47:49 2019

@author: jiaxinsun
"""

from utility.image_processing import *
from utility.gesture import Gesture


acute_angle_count_to_gesture = [Gesture.rock, Gesture.scissor, Gesture.scissor, Gesture.unkown, Gesture.paper]


def detect_with_contours(roi):
    
    # detect gesture based on skin
    thresh_img = detect_body_skin(roi)

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
            
            defects_img, num_defects = get_defects_count(roi, largest_contour, defects)
        
            images = concatenate_images(thresh_img, defects_img)
            
            # if the number of defects <=5, return
            if num_defects <= 5:
                
                return acute_angle_count_to_gesture[num_defects], images

    images = concatenate_images(thresh_img, roi)
    return Gesture.unkown, images
