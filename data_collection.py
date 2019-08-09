import cv2
import imutils
import numpy as np
import pandas as pd
import pickle

if __name__ == '__main__':
    training_data = {}
    # get the reference to the webcam
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_EXPOSURE, -4)

    # region of interest (ROI) coordinates
    top, right, bottom, left = 10, 350, 225, 590

    # initialize num of frames
    image_num = 0

    start_recording = False

    # keep looping, until interrupted
    while(True):
        # get the current frame
        (grabbed, frame) = camera.read()
        if (grabbed == True):
            # # resize the frame
            frame = imutils.resize(frame, width=700)
            # flip the frame so that it is not the mirror view
            frame = cv2.flip(frame, 1)
            # clone the frame
            clone = frame.copy()
            # get the height and width of the frame
            (height, width) = frame.shape[:2]
            # get the ROI
            roi = frame[top:bottom, right:left]
            # draw the segmented hand
            cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

            if start_recording:
                training_data[gesture_name].append(roi)
                image_num += 1
                print(image_num)

            # display the frame with segmented hand
            cv2.imshow("Video Feed", clone)
            cv2.imshow("data", roi)


            # observe the keypress by the user
            keypress = cv2.waitKey(1) & 0xFF

            # if the user pressed "q", then stop looping
            if keypress == ord("q"):
                break
        
            if keypress == ord("s"):
                gesture_name = input('gesture_name: ')
                training_data[gesture_name] = training_data.get(gesture_name, [])
                start_recording = True

            if keypress == ord("v"):
                with open('./training_data.pickle', 'wb') as handle:
                    pickle.dump(training_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print('save data success')
                start_recording = False


        else:
            print("[Warning!] Error input, Please check your(camra Or video)")
            break
    
    camera.release()

# free up memory
cv2.destroyAllWindows()

