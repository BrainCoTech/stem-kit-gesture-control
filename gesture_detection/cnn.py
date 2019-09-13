from utility.neural_network.nn_function import * 
from utility.frame_read import read_frame
from utility.gesture_detection import preprocess_cnn_img
import cv2
import mxnet as mx

def cnn_method(camera, net = None):
    roi = read_frame(camera)
    img = preprocess_cnn_img(roi)
    result = predict_class(img, net)
    return result