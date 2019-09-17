import cv2
import pickle
from utility.gesture import Gesture
from utility.camera import read_frame
from utility.image_processing import preprocess_for_cnn

training_data = {}

_IMAGE_COUNT_FOR_EACH_CLASS = 10

should_quit = False


def handle_key_press():
    global should_quit
    
    keypress = cv2.waitKey(1) & 0xFF
    if keypress == ord('q'):
        should_quit = True


if __name__ == "__main__":
    img_count = 0
    gesture_data_label = Gesture.paper  # Start collecting data with paper gesture

    camera = cv2.VideoCapture(0)

    while not should_quit:

        roi = read_frame(camera)
        handle_key_press()

        img = preprocess_for_cnn(roi)

        if img_count == 0:
            input("Press any key to collect data for:" + gesture_data_label.name)
            print('Collecting data for:' + gesture_data_label.name)

        img_count += 1
        
        if img_count == _IMAGE_COUNT_FOR_EACH_CLASS:
            if gesture_data_label == Gesture.scissor:  # Last gesture data set to collect
                with open('training_data.pickle', 'wb') as handle:
                    pickle.dump(training_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print('Successfully Saving Data')
                break

            gesture_data_label = Gesture(gesture_data_label.value + 1)
            img_count = 0
        
        else:
            training_data[gesture_data_label.name] = training_data.get(gesture_data_label.name, [])
            training_data[gesture_data_label.name].append(img)
            print(img_count)

    camera.release()
    cv2.destroyAllWindows()
