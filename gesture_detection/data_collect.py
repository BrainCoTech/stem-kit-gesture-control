import cv2
import pickle
from enum import Enum
from utility.frame_read import read_frame
from utility.gesture_detection import preprocess_cnn_img

label = ['paper', 'rock', 'scissor']
training_data = {label[0]:[], label[1]:[], label[2]:[]}
text = ['Press any key to collect PAPER data',
        'Press any key to collect ROCK data',
        'Press any key to collect SCISSOR data']
img_num_base = 20

def collect_data():
    img_num = 0
    should_quit = False
    camera = cv2.VideoCapture(0)
    while(not should_quit):
        roi = read_frame(camera)
        img = preprocess_cnn_img(roi)
        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord('q'):
            should_quit = True
        elif img_num%img_num_base == 0:
            input(text[int(img_num/img_num_base)])
        elif img_num//img_num_base == 0:
            training_data[label[0]].append(img)
            print('Collecting SCISSOR data: {}'.format(img_num))
        elif img_num//img_num_base == 1:
            training_data[label[1]].append(img)
            print('Collecting ROCK data: {}'.format(img_num-img_num_base))
        elif img_num//img_num_base == 2:
            training_data[label[2]].append(img)
            print('Collecting PAPER data: {}'.format(img_num-img_num_base*2))
        img_num += 1
        if img_num == 3*img_num_base:
            with open('./utility/neural_network/training_data.pickle', 'wb') as handle:
                pickle.dump(training_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('Successfully Saving Data')
            should_quit = True      
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    collect_data()