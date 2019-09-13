import cv2
import pickle
from utility.frame_read import read_frame
from utility.gesture_detection import preprocess_cnn_img


training_data = {'scissor':[], 'rock':[], 'paper':[]}
img_num_base = 100
label = ['scissor', 'rock', 'paper']
text = ["Press any key to start collecting SCISSOR gesture.", 
        "Press any key to start collecting ROCK gesture.", 
        "Press any key to start collecting PAPER gesture."]

def collect_data():
    img_num = 0
    camera = cv2.VideoCapture(0)
    while(True):
        roi = read_frame(camera)
        img = preprocess_cnn_img(roi)
        if img_num%img_num_base == 0:
            input(text[int(img_num/100)])
        elif img_num//img_num_base == 0:
            training_data[label[0]].append(img)
            print('Collecting SCISSOR data: {}'.format(img_num))
        elif img_num//img_num_base == 1:
            training_data[label[1]].append(img)
            print('Collecting ROCK data: {}'.format(img_num-img_num_base))
        elif img_num//img_num_base == 2:
            training_data[label[2]].append(img)
            print('Collecting PAPER data: {}'.format(img_num-img_num_base*2))
        elif img_num == 3*img_num_base:
            with open('./training_data.pickle', 'wb') as handle:
                pickle.dump(training_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('Successfully Saving Data')       
        img_num += 1
    camera.release()

if __name__ == "__main__":
    collect_data()