import cv2
import os
import pickle
from utility.gesture import Gesture
from utility.camera import read_frame
from utility.image_processing import preprocess_for_cnn

_src_dir = os.path.abspath(os.path.dirname(__file__))
_data_dir = os.path.join(_src_dir, "network_data", "training_data.pickle")

training_data = {}

_IMAGE_COUNT_FOR_EACH_CLASS = 30

start_recording = False


def handle_key_press():
    global start_recording
    keypress = cv2.waitKey(1) & 0xFF
    if keypress == ord('c'):
        start_recording = True


if __name__ == "__main__":

    img_count = 0
    camera = cv2.VideoCapture(0)

    while img_count < _IMAGE_COUNT_FOR_EACH_CLASS*3:

        roi = read_frame(camera)
        img = preprocess_for_cnn(roi)
        cv2.imshow('collect_img', img)

        # determine the wanted gesture based on img_count
        gesture_data_label = Gesture(img_count//_IMAGE_COUNT_FOR_EACH_CLASS)
        cv2.putText(roi, gesture_data_label.name,(5,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)


        # time to change gesture
        if img_count % _IMAGE_COUNT_FOR_EACH_CLASS == 0:
            cv2.putText(roi, "press c to continue",(5,50), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 1, cv2.LINE_AA)
            start_recording = False

        handle_key_press()

        if start_recording:
            # record gesture and update the img_count
            print(gesture_data_label)
            training_data[gesture_data_label.name] = training_data.get(gesture_data_label.name, [])
            training_data[gesture_data_label.name].append(img)
            img_count += 1

        cv2.imshow("roi", roi)

    # saving data
    with open(_data_dir, 'wb') as handle:
        pickle.dump(training_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Successfully Saving Data')

    camera.release()
    cv2.destroyAllWindows()
