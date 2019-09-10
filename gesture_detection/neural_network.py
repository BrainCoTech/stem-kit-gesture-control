import mxnet as mx
from mxnet.gluon import nn
import numpy as np
from imgaug import augmenters as iaa

random_times =  5
epoch = 1500
learning_rate = 0.0001

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
            
            self.features.add(_make_output(hidden_units=[]))
            self.output = nn.Dense(classes)

    def hybrid_forward(self, F, x):
        feature_vec = self.features(x)
        x = self.output(feature_vec)
        return x

def augmentation(img):
    seq1 = iaa.Sequential([
        iaa.Crop(px=(10, 50)), # crop images from each side by 0 to 16px (randomly chosen)
    ])
    seq3 = iaa.Sequential([
        iaa.Affine(rotate=(-45, 45)), 
    ])
    seqs = [seq1, seq3]
    for _ in range(random_times):
        for seq in seqs:
            aug_img = seq.augment_image(img)
    return img

def generate_data(training_data):
    scissor = np.array(training_data[label[0]])
    rock = np.array(training_data[label[1]])
    paper = np.array(training_data[label[2]])
    x = np.vstack((scissor, rock, paper))
    for i in range(x.shape[0]):
        x[i] = augmentation(x[i])
    onehot_scissor = np.array([1, 0, 0]).reshape(1,3)
    y_scissor = onehot_scissor.repeat(scissor.shape[0], axis=0)
    onehot_rock = np.array([0, 1, 0]).reshape(1,3)
    y_rock = onehot_rock.repeat(rock.shape[0], axis=0)
    onehot_paper = np.array([0, 0, 1]).reshape(1,3)
    y_paper = onehot_paper.repeat(paper.shape[0], axis=0)
    y = np.vstack((y_scissor, y_rock, y_paper))
    train = mx.gluon.data.ArrayDataset(x, y)
    train_data = mx.gluon.data.DataLoader(train, batch_size=1, last_batch='discard', shuffle=True)
    return train_data

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
    loss_function = mx.gluon.loss.SoftmaxCrossEntropyLoss()

    for e in range(epoch):
        # training model
        for i, (x,y) in enumerate(train_data):
            x = x.as_in_context(ctx)
            y = y.as_in_context(ctx)
            with mx.autograd.record():        
                output = net(x)
                l = loss(output, y)
            l.backward()            
            trainer.step(1)
            epoch_loss += l.mean()
        # train accuracy
        n = 0
        for x,y in train_data:
            output=net(x)
            if np.argmax(output, axis=1) ==  np.argmax(y, axis=1):
                n = n + 1
            acc = np.around(n/len(train_data), decimals=4)
    return net, acc



