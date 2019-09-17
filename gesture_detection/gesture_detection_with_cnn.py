
import mxnet as mx
import numpy as np
import os 
import cv2
import warnings
from utility.camera import read_frame
from utility.image_processing import preprocess_for_cnn

_src_dir = os.path.abspath(os.path.dirname(__file__))

network_symbol_path = os.path.join(_src_dir, "network_data", "trained_network-symbol.json")
network_params_path = os.path.join(_src_dir, "network_data", "trained_network-0000.params")

if os.path.exists(network_symbol_path) and os.path.exists(network_params_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = mx.gluon.SymbolBlock.imports(network_symbol_path, ['data'], network_params_path)


def predict_with_cnn(roi):
    # Preprocess image for CNN
    img = preprocess_for_cnn(roi)

    # Get output using mxnet
    outputs = mx.nd.softmax(net(mx.nd.array(img.reshape(1,1,128,128)))).asnumpy()
    
    # Use the largest output value as the prediction result
    gesture_index = np.argmax(outputs[0])

    return gesture_index, img
