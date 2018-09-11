# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 15:51:14 2018

@author: Sirius

change from original resnet_v2 change
"""

'''
Model Structure: resnet_34
1.   7X7,16/2

2.   3X3,64  block
3.   3X3,64  block
4.   3X3,64  block

5.   3X3,128/2  block
6.   3X3,128  block
7.   3X3,128  block
8.   3X3,128  block

9.   3X3,256/2  block
10.   3X3,256  block
11.   3X3,256  block
12.   3X3,256  block
13.   3X3,256  block
14.   3X3,256  block

15.   3X3,512/2  block
16.   3X3,512  block
17.   3X3,512  block

18.   average pooling
19.   fc200
20.   softmax
'''

'''
Note:
BN: added between conv and it's relu
filter number change: 1X1 conv
BN层是否也要参数共享？要

训练不同版本：
Version1:
   每一个building block中重复利用卷积权重
   按照ImageNet的卷积层设计，但是filter个数按照cifar10设计
   每一个unit中有两个3X3的卷积层，不按bottleneck的1X1,3X3,1X1的设计
   原始版本，conv-bn-relu，不按照pre-activation的设计   

'''

# 文章中cifar10的训练参数
BATCH_SIZE = 128
LR_DECAY = [32000,48000]
NUM_STEPS = 64000
WEIGHT_DECAY = 0.0001
MOMENTUM = 0.9

REPEAT_NUM = 5  # n: 每个block中卷积层重复的次数
FILTER = [64,128,256,512]

import tensorflow as tf
import numpy as np

def batch_norm(inputs, is_training, reuse = False):
  # training: GraphKeys.TRAINABLE_VARIABLES; axis = 3: channel last
  # batch_norm: axis=3--> channels_last, axis=1 --> channels_first
  return tf.layers.batch_normalization(
         inputs=inputs, axis = 1, momentum=0.997, epsilon=1e-5, center=True,
         scale=True, training=is_training, fused=reuse) 


def dense(inputs, units, name=None):
    """3x3 conv layer: ReLU + He initialization"""
    # He initialization: normal dist with stdev = sqrt(2.0/fan-in)
    # fan_in: number of input units, in fully connected layers
    stddev = np.sqrt(2 / int(inputs.shape[1])) 
    inputs = tf.layers.dense(inputs, units, activation=tf.nn.relu,
                            kernel_initializer=tf.random_normal_initializer(stddev=stddev),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0),
                            name=name)
    return tf.identity(inputs, name)


def building_block(inputs, is_training, filters, name,  kernel_size, 
                   first_layer_strides = 1,data_format='channels_first', reuse = False):
    # tf.layers.conv2d: default, padding = 'valid', stride = 1
    # ResNet paper (projection B): go cross feature maps of different size, stide = 2
    #                              therefore decrease resolution by 2
    # params:
    # first_layer_strides: first block does not downsampling, stride = 1, otherwise strides = 2
    #                      查看这个builing block是否需要降维度
    # print('Inside building block ==================')
    # print('inputs',inputs)
    # print('first_layer_strides',first_layer_strides)

    with tf.variable_scope(name) as scope: 
        with tf.variable_scope('shortcut') as scope:
            shortcuts = inputs
            shortcuts = tf.layers.conv2d(inputs, filters = filters, kernel_size = 1, padding = 'same',
                                         strides = first_layer_strides, data_format=data_format)
            shortcuts = batch_norm(shortcuts, is_training, reuse)
            # print('inside building block, shortcuts', shortcuts)

        with tf.variable_scope('sub1') as scope:
            stddev = np.sqrt(2 / (np.prod([kernel_size, kernel_size]) * int(inputs.shape[3])))
            inputs = tf.layers.conv2d(inputs, filters=filters, kernel_size = kernel_size ,padding='same', 
                                     strides = first_layer_strides,
                                     kernel_initializer=tf.random_normal_initializer(stddev=stddev),
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0),
                                     data_format=data_format)
            inputs = batch_norm(inputs, is_training, reuse)
            inputs = tf.nn.relu(inputs)
            # print('inside building block, sub1 inputs', inputs)
        
        with tf.variable_scope('sub2') as scope:
            inputs = tf.layers.conv2d(inputs, filters=filters, kernel_size=kernel_size,padding='same', 
                                     kernel_initializer=tf.random_normal_initializer(stddev=stddev),
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0),
                                     data_format=data_format)
            inputs = batch_norm(inputs, is_training, reuse)
            # print('inside building block, sub2 inputs', inputs)
        
        with tf.variable_scope('join') as scope:
             inputs = tf.nn.relu(inputs + shortcuts)
             # print('inside building block, join inputs', inputs)

    return inputs

# sirius: 在Input换到channel first 时，conv_2d也要换
def resnet(images, is_training):
    inputs = tf.cast(images, tf.float32)
    # inputs = (img - 128.0)/128.0 # ??? sirius: cetering the image [0,255]
    # print('original inputs shape', inputs.shape)
    # tf.summary.histogram('img', inputs)
    inputs = tf.transpose(inputs, [0, 3, 1, 2])  # performance boost on GPU: channel_last to channel_first
    print('inputs shape', inputs.shape)
    # (N, 3, 56, 56)
    
    # 第一个卷积层默认strides=1
    with tf.variable_scope('conv1') as scope:
        inputs = tf.layers.conv2d(inputs, filters = FILTER[0], kernel_size = 7, 
                                  padding='same', data_format='channels_first') # 64 7X7
        inputs = batch_norm(inputs, is_training)
        inputs = tf.nn.relu(inputs)
        print('conv1 inputs shape', inputs.shape)
        # (N, 64, 56, 56)
    
    with tf.variable_scope('conv2') as scope:
        inputs = building_block(inputs, is_training, filters = FILTER[0], name = 'layer1',
                                kernel_size = 3, first_layer_strides = 1, data_format='channels_first')
        inputs = building_block(inputs, is_training, filters = FILTER[0], name = 'layer2',
                                kernel_size = 3, first_layer_strides = 1, data_format='channels_first')
        inputs = building_block(inputs, is_training, filters = FILTER[0], name = 'layer3',
                                kernel_size = 3, first_layer_strides = 1, data_format='channels_first')
        print('conv2 inputs shape', inputs.shape)
        # (N, 64, 56, 56)
    
    with tf.variable_scope('conv3') as scope:
        inputs = building_block(inputs, is_training, filters = FILTER[1], name = 'layer1',
                                kernel_size = 3, first_layer_strides = 2, data_format='channels_first')
        inputs = building_block(inputs, is_training, filters = FILTER[1], name = 'layer2',
                                kernel_size = 3, first_layer_strides = 1, data_format='channels_first')
        inputs = building_block(inputs, is_training, filters = FILTER[1], name = 'layer3',
                                kernel_size = 3, first_layer_strides = 1, data_format='channels_first')
        print('conv3 inputs shape', inputs.shape)
        # (N, 128, 28, 28)
    
    with tf.variable_scope('conv4'):
        inputs = building_block(inputs, is_training, filters = FILTER[2], name = 'layer1',
                                kernel_size = 3, first_layer_strides = 2,data_format='channels_first')
        inputs = building_block(inputs, is_training, filters = FILTER[2], name = 'layer2',
                                kernel_size = 3, first_layer_strides = 1,data_format='channels_first')
        inputs = building_block(inputs, is_training, filters = FILTER[2], name = 'layer3',
                                kernel_size = 3, first_layer_strides = 1,data_format='channels_first')
        inputs = building_block(inputs, is_training, filters = FILTER[2], name = 'layer4',
                                kernel_size = 3, first_layer_strides = 1,data_format='channels_first')
        inputs = building_block(inputs, is_training, filters = FILTER[2], name = 'layer5',
                                kernel_size = 3, first_layer_strides = 1,data_format='channels_first')
        print('conv4 inputs shape', inputs.shape)
        # (N, 256, 14, 14)
    
    with tf.variable_scope('conv5') as scope:
        inputs = building_block(inputs, is_training, filters = FILTER[3],  name = 'layer1',
                                kernel_size = 3, first_layer_strides = 2, data_format='channels_first')
        inputs = building_block(inputs, is_training, filters = FILTER[3], name = 'layer2',
                                kernel_size = 3, first_layer_strides = 1, data_format='channels_first')
        print('conv5 inputs shape', inputs.shape)
        # (N, 512, 7, 7)

    # global pooling layer
    # inputs = tf.reduce_mean(inputs, [1,2], keepdims = True) # keepdims: If true, retains reduced dimensions with length 1.
    with tf.variable_scope('average_pooling') as scope:
        inputs = tf.reduce_mean(inputs, [2,3], keepdims = True)
        print('global pooling inputs shape', inputs.shape)
        # (N, 512)
    
    # fc layer
    with tf.variable_scope('fc') as scope:
        inputs = dense(tf.reshape(inputs, [-1, FILTER[3]]), 200, name='fc')
        print('fc inputs shape', inputs.shape)
        # (N, 200)
    
    return inputs

'''
# Test Example
data = tf.zeros((10, 64, 64, 3))

# 1X1 convolution with strides = 2
# output1 = tf.layers.conv2d(data, filters = 128, kernel_size =1, strides = 2, padding = 'valid') # (10,32,32,128)
# output2 = tf.layers.conv2d(data, filters = 128, kernel_size =1, strides = 2, padding = 'same')  # (10,32,32,128)
# print('output1,shape', output1.shape)
# print('output2,shape', output2.shape)

output = resnet(data, True)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    result=sess.run(ouput)
'''




