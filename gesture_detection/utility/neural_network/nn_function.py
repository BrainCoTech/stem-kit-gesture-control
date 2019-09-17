import numpy as np
import mxnet as mx
import warnings
import os 

symbol_file = './utility/neural_network/trained_network-symbol.json'
params_file = './utility/neural_network/trained_network-0000.params'

data_exist = os.path.exists('./utility/neural_network/training_data.pickle')
if data_exist:
    symbol_file = './utility/neural_network/trained_network-symbol.json'
    params_file = './utility/neural_network/trained_network-0000.params'

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    net = mx.gluon.SymbolBlock.imports(symbol_file, ['data'], params_file)

def predict_class(img):
    # put preprocessed image into CNN and get probability array in shape (1, number of class)
    outputs = mx.nd.softmax(net(mx.nd.array(img.reshape(1,1,128,128)))).asnumpy()
    # label index 
    cur_gesture_index = np.argmax(outputs[0])
    return cur_gesture_index
