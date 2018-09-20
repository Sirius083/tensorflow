# 数据预处理：按照resnet原始文章
# 训练： 1. padding 4 pixel + 32X32 crop 2. horizontal flip
# 测试： 2. 直接使用32X32的测试数据
# lr: lr = 0.01 warm up training --> until the training error between 80%(400 ieration)

# cifar100: 100 classes 
#           600 images/each, 500 training, 100 testing
#           20  superclasses(coarse label)
# binary 格式：<1Xcoarse label><1Xfine label><3072 pixel>

# 师兄测试过每个epoch都打乱训练数据对结果影响不大

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

IMG_WIDTH = 32
IMG_HEIGHT = 32
IMG_DEPTH = 3
NUM_CLASS = 100

EPOCH_SIZE = 50000
EPOCH_SIZE_TRAIN = 45000 # 训练集的总样本
EPOCH_SIZE_VALI = 5000   # 验证集的总样本数
VALI_SIZE_PER_CLASS = 50

# 数据预处理阶段使用的函数 ==================================
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

    data = np.array(dicts['data'])
    label = np.array(dicts['fine_labels'])
    # print('train_data.shape', train_data.shape)
    # print('train_label.shape',train_label.shape)
    
    # label_to_name = meta['fine_label_names'] # 长度为 100 的list
    return data, label


def read_in_all_images(data_path,shuffle=True):
    data, label = read_data(data_path)
    # data and label sort array by index
    sort_index = np.argsort(label)
    label = label[sort_index]
    data = data[sort_index,...]

    data = data.reshape((len(data), IMG_HEIGHT * IMG_WIDTH, IMG_DEPTH), order='F') 
    data = data.reshape((len(data), IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH))
    data = data.astype(np.float32)
    
    # sampling without replacement
    # range(1,9): 包括 1，不包括 9
    # validation index
    sample_per_class = int(EPOCH_SIZE/NUM_CLASS)
    from random import sample
    from random import seed
    seed(1) # same seperation in each training time
    vali_index = []
    for i in range(NUM_CLASS):
        ind = sample(range(i*sample_per_class, (i+1)*sample_per_class), VALI_SIZE_PER_CLASS)
        vali_index = vali_index + ind
    vali_data = data[vali_index,...]
    vali_label = label[vali_index]
          
    # train index
    train_index = list(set(np.arange(EPOCH_SIZE)) - set(vali_index))
    train_data = data[train_index,...]
    train_label = label[train_index,...]
    
    # 需要打乱训练样本
    # 由于训练过程中也是选择batch数据集进行测试，需要打乱
    if shuffle is True:
        print ('Shuffling')
        order = np.random.permutation(len(train_label))
        train_data = train_data[order, ...]
        train_label = train_label[order]

        order_vali = np.random.permutation(EPOCH_SIZE_VALI)
        vali_data = vali_data[order_vali,...]
        vali_label = vali_label[order_vali]

    return train_data,train_label,vali_data,vali_label

def prepare_train_data(data_path, padding_size):
    train_data,train_label,vali_data,vali_label = read_in_all_images(data_path)
    # padding train data
    pad_width = ((0, 0), (padding_size, padding_size), (padding_size, padding_size), (0, 0))
    train_data = np.pad(train_data, pad_width=pad_width, mode='constant', constant_values=0)
    return train_data,train_label,vali_data,vali_label

def read_test_data(test_path):
    data, label = read_data(test_path)
    num_data = len(label)
    data = data.reshape((num_data, IMG_HEIGHT * IMG_WIDTH, IMG_DEPTH), order='F') 
    data = data.reshape((num_data, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH))
    data = data.astype(np.float32)
    return data, label

# 训练过程中的函数 ==================================
def random_crop_and_flip(batch_data, padding_size):
    cropped_batch = np.zeros(len(batch_data) * IMG_HEIGHT * IMG_WIDTH * IMG_DEPTH).reshape(
        len(batch_data), IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH)

    for i in range(len(batch_data)):
        x_offset = np.random.randint(low=0, high=2 * padding_size, size=1)[0]
        y_offset = np.random.randint(low=0, high=2 * padding_size, size=1)[0]
        cropped_batch[i, ...] = batch_data[i, ...][x_offset:x_offset+IMG_HEIGHT,
                      y_offset:y_offset+IMG_WIDTH, :]
        
        cropped_batch[i, ...] = horizontal_flip(image=cropped_batch[i, ...], axis=1)
    return cropped_batch



if __name__ == '__main__':
  data_dir = 'cifar100_data'
  data_path = 'cifar100_data/cifar-100-python/train'
  meta_path = 'cifar100_data/cifar-100-python/meta'
  test_path = 'cifar100_data/cifar-100-python/test'
  
  train_data,train_label,vali_data,vali_label = prepare_train_data(data_path,4)
  test_data, test_label = read_test_data(test_path)
