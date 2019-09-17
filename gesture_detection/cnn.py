from utility.neural_network.nn_function import predict_class 
from utility.frame_read import read_frame
from utility.gesture_detection import preprocess_cnn_img
import cv2
import mxnet as mx

def cnn_method(roi, net = None):
    img = preprocess_cnn_img(roi)
    result = predict_class(img)
    return result
