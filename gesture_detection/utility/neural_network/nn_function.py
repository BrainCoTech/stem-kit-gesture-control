import numpy as np
import mxnet as mx
import warnings

symbol_file = ''
params_file = ''

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    default_net = mx.gluon.SymbolBlock.imports(symbol_file, ['data'], params_file)

def predict_class(img, net):
    if not net:
        net = default_net
    # put preprocessed image into CNN and get probability array in shape (1, number of class)
    outputs = mx.nd.softmax(net(mx.nd.array(img.reshape(1,1,128,128)))).asnumpy()
    # label index 
    cur_gesture_index = np.argmax(outputs[0])
    return cur_gesture_index


