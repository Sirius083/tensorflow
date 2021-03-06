#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 有用函数

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"   

import numpy as np
import tensorflow as tf

# denseNet official(torch)数据预处理只保留了一位小数
cifar10_mean = np.array([125.3, 123.0, 113.9])
cifar10_std = np.array([63.0,  62.1,  66.7])
cifar100_mean = np.array([129.3, 124.1, 112.4])
cifar100_std = np.array([68.2 , 65.4, 70.4])

def get_graph(graph_pb_path):
    tf.reset_default_graph()
    config = tf.ConfigProto(allow_soft_placement = True)

    with tf.gfile.GFile(graph_pb_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
    # print('*******************************')
    print('load graph pb done')
    return graph


# cifar10 and cifar100 are cifar-python format
# test on mixnet pretrained model on cifar10

def cifar10_load_data(data_path, filename):
    # load a pickled data-file from the cifar10
    # params:
    # data_path: directory to store all data
    # filename: 'train', 'test'

    # data augmentation: 
    # 1. data / 128.0 - 1
    # 2. subtract per pixel mean

    import os
    import numpy as np
    import pickle

    if filename == 'test':
        file_path = os.path.join(data_path, 'cifar-10-batches-py', 'test_batch')

        with open(file_path, mode = 'rb') as file_:
            # It is important to set the encoding
            Data = pickle.load(file_, encoding = 'bytes')
        
        data = Data[b'data']         # (10000,3072)
        labels = np.array(Data[b'labels']) # (10000,)
        print("Data being loaded: "+ file_path)

    elif filename == 'train':
        
         cifar10_dir = os.path.join(data_path, 'cifar-10-batches-py')
         filenames = os.listdir(cifar10_dir)[1:6]

         for i in range(5):
            file_path = os.path.join(cifar10_dir,filenames[i])
            print('file_path', file_path)
            if i == 0:
               with open(file_path, mode = 'rb') as file_:
                    Data = pickle.load(file_, encoding = 'bytes')
                    data = Data[b'data']
                    labels = np.array(Data[b'labels'])
            else:
                 with open(file_path, mode = 'rb') as file_:
                    Data = pickle.load(file_, encoding = 'bytes')
                    data_batch = Data[b'data']
                    labels_batch = np.array(Data[b'labels'])
                    data = np.concatenate((data, data_batch), axis = 0)
                    labels = np.concatenate((labels, labels_batch), axis = 0)

    images = np.array(data, dtype = float)

    # reshape the array to 4-dimensions
    images = images.reshape([-1,3, 32, 32])

    # reorder the indicies of the array
    images = images.transpose([0,2,3,1])
        
    '''
    # data augmentation: subtract per pixel mean in dataset
    pp_mean = np.mean(images, axis=0)
    images = images -  pp_mean
    images = images / 128.0 - 1
    '''
    # images = images / 255.0
    
    return images, labels

'''
import numpy as np
data, label = cifar10_load_data(r'E:\Data', 'train')
# data, label = cifar10_load_data(r'E:\Data', 'test')
print('cifar10 mean', np.mean(data, (0,1,2)))
print('cifar10 std', np.std(data, (0,1,2)))  
'''


def cifar100_load_data(data_path, filename):
    import os
    import numpy as np
    import pickle 
    # data_path:  r'E:\Data'
    # filename:  'train', 'test'
   
    file_path = os.path.join(data_path, 'cifar-100-python', filename)
    with open(file_path, mode = 'rb') as file_:
        # It is important to set the encoding
        data = pickle.load(file_, encoding = 'bytes')
    images = data[b'data']                 
    labels = np.array(data[b'fine_labels'])
    print("Data being loaded: "+ file_path)
    
    images = np.array(images, dtype = float)
    
    # reshape the array to 4-dimensions
    images = images.reshape([-1,3, 32, 32])
    
    # reorder the indicies of the array
    images = images.transpose([0,2,3,1])
    
    # images = images / 255.0

    return images, labels



if __name__ == '__main__':
    # data, label = cifar100_load_data(r'E:\Data', 'train')
    data, label = cifar100_load_data(r'E:\Data', 'test')
    # print('data', data.shape)   # (10000,32,32,3)
    # print('label', label.shape) # (10000,)
    data = (data - cifar100_mean)/cifar100_std
    print('data.shape', data.shape)

    # print('cifar100 mean', np.mean(data, (0,1,2)))
    # print('cifar100 std', np.std(data, (0,1,2)))  



'''
# cifar100 names
# cifar100 fine and coarse mapping are in the webpage
import pickle
import os
meta_path = os.path.join(data_path, 'cifar-100-python', 'meta')
with open(meta_path, 'rb') as fo:
    meta = pickle.load(fo, encoding='bytes')
# the name order is class order
fine_label_names = [t.decode('utf8') for t in meta[b'fine_label_names']]
coarse_label_names = [t.decode('utf8') for t in meta[b'coarse_label_names']]
'''

'''
# 注：原代码中的顺序是
# 1. images - pp_mean; 2. images/128.0 - 1.0
# 我的预处理代码是：
# 1. images/128.0 - 1; 2. images - pp_mean;
# show images
import scipy.misc
rgb = scipy.misc.toimage(images[1000])
rgb
'''
    
