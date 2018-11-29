# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 09:41:31 2018

@author: Sirius

检查两个共享权重的卷积网络在tensorboard中的可视化画图
注：tensorboard画图需要给每一个运算起名字，比如relu, 如果没起名字，会将relu聚合成为一个点，图片混乱
    不能给batch_normalization乱改名字，否则画出的图也很混乱
"""

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

training = True
timeout = 2

import tensorflow as tf


def BRC(inputs, filters, kernel_size, strides, training, name, reuse = None, is_bn = True):
    # inputs:       (batch_size, feature_size, feature_size, filter)
    # filters:      通道数
    # kernel_size: 卷积核大小
    # strides:     卷积步长
    # training:    training/validation/evaluation
    # name:        在当前 variable_scope下的名称
    # iternum:     是当前重复层的第几层
    # reuse:       是否需要重复使用卷积层
    # bn:          是否需要定义新的 batch_normalization层
    inputs = tf.layers.conv2d(
        inputs, filters = filters, kernel_size = kernel_size, strides = strides, padding = "SAME",
        use_bias = False, reuse = reuse, name = name, kernel_initializer = tf.contrib.layers.xavier_initializer())
    inputs = tf.nn.relu(inputs, name = name)
    # 每一个自己定义的卷积层带上一组batchnorm参数（一组4个）
    # 没有自己定义的不带batchnorm参数
    if is_bn:
       inputs = tf.layers.batch_normalization(
            inputs=inputs, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, 
            training=training, fused=True)
    return tf.identity(inputs, name)

# pre_conv: (N, 32, 32, 16)
inputs = tf.zeros((10, 32,32,3))

with tf.variable_scope('pre'):
     inputs = BRC(inputs, filters = 16, kernel_size = 3, strides = 1, training = training, name = 'conv')
     print('pre inputs', inputs)
    
with tf.variable_scope('h1'):
     shortcut = inputs
     inputs = BRC(inputs, filters = 16, kernel_size = 3, strides = 1, training = training, name = 'conv1')
     print('h1 conv1 inputs', inputs)
     inputs = BRC(inputs, filters = 16, kernel_size = 3, strides = 1, training = training, name = 'conv2')
     print('h1 conv2 inputs', inputs)
     inputs = BRC(inputs, filters = 16, kernel_size = 3, strides = 1, training = training, name = 'conv3')
     print('h1 conv3 inputs', inputs)
     
     for i in range(1,timeout):
         inputs = BRC(inputs, filters = 16, kernel_size = 3, strides = 1, training = training, name = 'conv1', reuse = True, is_bn = False)
         print('h1 conv1 repeat inputs', inputs)
         inputs = BRC(inputs, filters = 16, kernel_size = 3, strides = 1, training = training, name = 'conv2', reuse = True, is_bn = False)
         print('h1 conv1 repeat inputs', inputs)
         inputs = BRC(inputs, filters = 16, kernel_size = 3, strides = 1, training = training, name = 'conv3', reuse = True, is_bn = False)
         print('h1 conv1 repeat inputs', inputs)
     inputs = inputs + shortcut


################################################################################
# 输出所有变量和在tensorboard中画出图
################################################################################
allvars = tf.global_variables()
for var in allvars:
        print(var)
# draw tensorboard graph to see the data flow
init = tf.initializers.global_variables()
sess = tf.Session()
sess.run(init)

test_graph_path = r'E:\resnet\models-master\official\resnet_cifar10_recurrent_conv\test_graph'
print('current working directory',test_graph_path)
summary_writer = tf.summary.FileWriter(test_graph_path, sess.graph)

