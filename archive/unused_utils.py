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

# calculate center coordinate of contour
def get_center(largecont):
    M = cv2.moments(largecont)
    center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
    return center