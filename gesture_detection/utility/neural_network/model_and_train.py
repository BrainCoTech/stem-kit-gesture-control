import numpy as np
import pandas as pd
import mxnet as mx
from mxnet.gluon import nn, HybridBlock
from data_prepare import prepare_data

epoch = 100
learning_rate = 0.0001
filename = 'trained_network'

# Net
class inception_network(HybridBlock):
    def __init__(self, classes=3, **kwargs):
        super(inception_network, self).__init__(**kwargs)
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(nn.Conv2D(use_bias=True, channels = 4, 
                                        kernel_size = (5,5), strides = (1,1), 
                                        padding = (1,1), activation='relu'))
            self.features.add(nn.AvgPool2D(pool_size=3, strides=3, padding=1))
            
            self.features.add(nn.Conv2D(use_bias=True, channels = 8, 
                                        kernel_size = (3,3), strides = (1,1), 
                                        padding = (1,1), activation='relu'))
            self.features.add(nn.AvgPool2D(pool_size=3, strides=3, padding=1))
            
            self.features.add(nn.Conv2D(use_bias=True, channels = 16, 
                                        kernel_size = (3,3), strides = (1,1), 
                                        padding = (1,1), activation='relu'))
            self.features.add(nn.AvgPool2D(pool_size=3, strides=3, padding=1))
            
            # self.features.add(_make_output(hidden_units=[]))
            self.output = nn.Dense(classes)

    def hybrid_forward(self, F, x):
        feature_vec = self.features(x)
        x = self.output(feature_vec)
        return x

def net_summary():
    net = inception_network()
    ctx = mx.gpu(0) if mx.context.num_gpus() > 0 else mx.cpu(0)
    net.initialize(mx.init.Xavier(), ctx=ctx)
    data_input = mx.nd.array(np.random.randint(low=0, high=20, size=(1,1,128,128)))
    net.summary(data_input)

def train_model(train_data):
    net = inception_network()
    net.hybridize()
    ctx = mx.gpu(0) if mx.context.num_gpus() > 0 else mx.cpu(0)
    net.initialize(mx.init.Xavier(), ctx=ctx)
    trainer = mx.gluon.Trainer(
        params=net.collect_params(),
        optimizer='Adam',
        optimizer_params={'learning_rate': learning_rate},
    )
    loss_function = mx.gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False)
    for e in range(epoch):
        n = 0
        epoch_loss = 0
        # training model
        for i, (x,y) in enumerate(train_data):
            x = x.as_in_context(ctx)
            y = y.as_in_context(ctx)
            with mx.autograd.record():           
                output = net(x)
                l = loss_function(output, y)
            l.backward()            
            trainer.step(1)
            epoch_loss += l.mean()
        # train accuracy
        for x,y in train_data:
            output=net(x)
            if np.argmax(output, axis=1) ==  np.argmax(y, axis=1):
                n = n + 1
            acc = np.around(n/len(train_data), decimals=4)
    net.export(filename)
    return net, acc

if __name__ == "__main__":
    data = pd.read_pickle('./training_data.pickle')
    train_data = prepare_data(data)
    net, acc = train_model(train_data)


