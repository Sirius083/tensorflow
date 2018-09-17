# 数据预处理：按照resnet原始文章
# 训练： 1. padding 4 pixel + 32X32 crop 2. horizontal flip
# 测试： 2. 直接使用32X32的测试数据
# lr: lr = 0.01 warm up training --> until the training error between 80%(400 ieration)

# cifar100: 100 classes 
#           600 images/each, 500 training, 100 testing
#           20  superclasses(coarse label)
# binary 格式：<1Xcoarse label><1Xfine label><3072 pixel>

# 读 cifar100 数据
# Note: 1. validation 直接是 test 数据
# Note: 2. 每个batch是从所有数据中随机抽取的
#==============================================================================
import tarfile
from six.moves import urllib
import sys
import numpy as np
import _pickle as pickle
import os
import cv2
# import tensorflow as tf


data_dir = 'cifar100_data'
data_path = 'cifar100_data/cifar-100-python/train'
meta_path = 'cifar100_data/cifar-100-python/meta'
test_path = 'cifar100_data/cifar-100-python/test'
# DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'

IMG_WIDTH = 32
IMG_HEIGHT = 32
IMG_DEPTH = 3
NUM_CLASS = 100

EPOCH_SIZE = 50000

def horizontal_flip(image, axis):
    '''
    Flip an image at 50% possibility
    :param image: a 3 dimensional numpy array representing an image
    :param axis: 0 for vertical flip and 1 for horizontal flip
    :return: 3D image after flip
    '''
    flip_prop = np.random.randint(low=0, high=2) # sirius:[inclusive, exclusive]
    if flip_prop == 0:
        image = cv2.flip(image, axis)

    return image

def read_data(data_path):
    # meta: class 标号 与 数字
    with open(data_path, 'rb') as fo:
         dicts = pickle.load(fo,encoding='latin1')
         
         
    # with open(meta_path, 'rb') as fo:
    #      meta = pickle.load(fo,encoding='latin1')

         
    train_data = np.array(dicts['data'])
    train_label = np.array(dicts['fine_labels'])
    # print('train_data.shape', train_data.shape)
    # print('train_label.shape',train_label.shape)
    
    # label_to_name = meta['fine_label_names'] # 长度为 100 的list
    return train_data, train_label


def read_in_all_images(data,label,shuffle=True):
    num_data = len(label)
    data = data.reshape((num_data, IMG_HEIGHT * IMG_WIDTH, IMG_DEPTH), order='F') 
    data = data.reshape((num_data, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH))

    if shuffle is True:
        print ('Shuffling')
        order = np.random.permutation(num_data)
        data = data[order, ...]
        label = label[order]
    
    data = data.astype(np.float32)
    return data, label


def random_crop_and_flip(batch_data, padding_size):
    # 训练过程中每个batch处理一次
    cropped_batch = np.zeros(len(batch_data) * IMG_HEIGHT * IMG_WIDTH * IMG_DEPTH).reshape(
        len(batch_data), IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH)

    for i in range(len(batch_data)):
        x_offset = np.random.randint(low=0, high=2 * padding_size, size=1)[0]
        y_offset = np.random.randint(low=0, high=2 * padding_size, size=1)[0]
        cropped_batch[i, ...] = batch_data[i, ...][x_offset:x_offset+IMG_HEIGHT,
                      y_offset:y_offset+IMG_WIDTH, :]
        
        cropped_batch[i, ...] = horizontal_flip(image=cropped_batch[i, ...], axis=1)
    return cropped_batch


def prepare_train_data(padding_size):
    data, label = read_data(data_path)
    data, label = read_in_all_images(data, label)
    # 训练数据增加了 padding 加上剪裁这一步
    pad_width = ((0, 0), (padding_size, padding_size), (padding_size, padding_size), (0, 0))
    data = np.pad(data, pad_width=pad_width, mode='constant', constant_values=0)
    return data, label


def read_validation_data():
    # validation data 需要打乱，由于每次只验证一个 batch 上的
    data, label = read_data(test_path)
    validation_array, validation_labels = read_in_all_images(data, label)

    return validation_array, validation_labels

if __name__ == '__main__':
   train_data = prepare_train_data(4)
   # test_data = read_validation_data()
