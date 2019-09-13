import numpy as np
from imgaug import augmenters as iaa
import mxnet as mx

random_times = 1
label = ['scissor', 'rock', 'paper']

def augmentation(img):
    seq = iaa.Sequential([
        iaa.Crop(px=(10, 50)), # crop images from each side by 0 to 16px (randomly chosen)
    ])
    aug_img = seq.augment_image(img)
    return aug_img

def prepare_data(data):
    x_train, y_train = [], []
    for i in range(len(label)):
        y = np.zeros((1, len(label)))
        y[0,i] = 1
        cur_label = label[i]
        cur_x = np.array(data[cur_label])
        h,w = cur_x[0].shape
        for j in range(cur_x.shape[0]):
            x_train.append(cur_x[j].reshape(1,h,w))  
            y_train.append(y)

            for _ in range(random_times):
                aug_img = augmentation(cur_x[j])
                x_train.append(aug_img.reshape(1,h,w))
                y_train.append(y)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    y = y_train.reshape((y_train.shape[0], len(label)))
    train = mx.gluon.data.ArrayDataset(x_train.astype('float32'), y.astype('float32'))
    train_data = mx.gluon.data.DataLoader(train, batch_size=1, last_batch='discard', shuffle=True)
    return train_data
    