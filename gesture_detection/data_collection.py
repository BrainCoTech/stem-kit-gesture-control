import cv2
import imutils
import numpy as np
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

        clone, roi = frame_read.read_frame(camera, top, right, bottom, left)

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
    
    camera.release()

# free up memory
cv2.destroyAllWindows()

