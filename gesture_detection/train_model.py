import numpy as np
import pandas as pd
import mxnet as mx
import os
from mxnet.gluon import nn, HybridBlock
from imgaug import augmenters as iaa
from utility.gesture import Gesture


epoch = 100
learning_rate = 0.0001
filename = "trained_network"
random_times = 1
class_num = 3

_src_dir = os.path.abspath(os.path.dirname(__file__))
_network_dir = os.path.join(_src_dir, "network_data", filename)
_data_dir = os.path.join(_src_dir, "network_data", "training_data.pickle")

def augmentation(img):
    seq = iaa.Sequential([
        iaa.Crop(px=(10, 50)),  # crop images from each side by 0 to 16px (randomly chosen)
    ])
    aug_img = seq.augment_image(img)
    return aug_img


def prepare_data(data):
    x_train, y_train = [], []
    for i in range(class_num):
        y = np.zeros((1, class_num))
        y[0, i] = 1
        cur_label = Gesture(i).name
        cur_x = np.array(data[cur_label])
        h, w = cur_x[0].shape
        for j in range(cur_x.shape[0]):
            x_train.append(cur_x[j].reshape(1, h, w))
            y_train.append(y)

            for _ in range(random_times):
                aug_img = augmentation(cur_x[j])
                x_train.append(aug_img.reshape(1, h, w))
                y_train.append(y)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    y = y_train.reshape((y_train.shape[0], class_num))
    train = mx.gluon.data.ArrayDataset(x_train.astype('float32'), y.astype('float32'))
    gluon_data = mx.gluon.data.DataLoader(train, batch_size=1, last_batch='discard', shuffle=True)
    return gluon_data


# Net
class Network(HybridBlock):
    def __init__(self, classes=3, **kwargs):
        super(Network, self).__init__(**kwargs)
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(nn.Conv2D(use_bias=True, channels=4,
                                        kernel_size=(5, 5), strides=(1, 1),
                                        padding=(1, 1), activation='relu'))
            self.features.add(nn.AvgPool2D(pool_size=3, strides=3, padding=1))
            
            self.features.add(nn.Conv2D(use_bias=True, channels=8,
                                        kernel_size=(3, 3), strides=(1, 1),
                                        padding=(1, 1), activation='relu'))
            self.features.add(nn.AvgPool2D(pool_size=3, strides=3, padding=1))
            
            self.features.add(nn.Conv2D(use_bias=True, channels=16,
                                        kernel_size=(3, 3), strides=(1, 1),
                                        padding=(1, 1), activation='relu'))
            self.features.add(nn.AvgPool2D(pool_size=3, strides=3, padding=1))
            
            # self.features.add(_make_output(hidden_units=[]))
            self.output = nn.Dense(classes)

    def hybrid_forward(self, F, x):
        feature_vec = self.features(x)
        x = self.output(feature_vec)
        return x


def print_network():
    net = Network()
    ctx = mx.gpu(0) if mx.context.num_gpus() > 0 else mx.cpu(0)
    net.initialize(mx.init.Xavier(), ctx=ctx)
    data_input = mx.nd.array(np.random.randint(low=0, high=20, size=(1, 1, 128, 128)))
    net.summary(data_input)


def train_model(data):
    net = Network()
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
        for i, (x, y) in enumerate(data):
            x = x.as_in_context(ctx)
            y = y.as_in_context(ctx)
            with mx.autograd.record():           
                output = net(x)
                loss = loss_function(output, y)

            loss.backward()
            trainer.step(1)
            epoch_loss += loss.mean()

        # train accuracy
        for x, y in data:
            output = net(x)
            if np.argmax(output, axis=1) == np.argmax(y, axis=1):
                n = n + 1
        accuracy = np.around(n/len(data), decimals=4)
        print("Training accuracy:", accuracy)
    net.export(_network_dir)
    return net


if __name__ == "__main__":
    pickle_data = pd.read_pickle(_data_dir)
    train_data = prepare_data(pickle_data)
    _ = train_model(train_data)
