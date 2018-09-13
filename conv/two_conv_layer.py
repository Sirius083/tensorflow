# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 20:56:11 2018

@author: Sirius

two conv layer with different inputs share weights
loss with respect to different layers has different gradients
"""

# Question: https://stackoverflow.com/questions/42862300/tensorflow-reuse-variable-with-tf-layers-conv2d
# variable get reused in the two convo  
# two layers share the name but not share the computation  
import tensorflow as tf
import numpy as np 

# 神经网络的每一层
# x = tf.random_normal(shape=[10, 32, 32, 3]) # 训练样本
x = np.random.normal(size = 10*32*32*3).reshape([10,32,32,3])
y = np.random.randint(low=1,high=20,size=10)
labels = tf.convert_to_tensor(y, dtype=tf.int32)

inputs = tf.placeholder(tf.float32, shape = [10,32,32,3])
# Note: reuse=tf.AUTO_REUSE, first time False, second time True
# 如果后面加上 use_bias 就不初始化 bias
conv1 = tf.layers.conv2d(inputs, 16, [3, 3], padding='SAME', reuse=None, name='conv1',use_bias = False)

conv2 = tf.layers.conv2d(conv1, 16, [3, 3], padding='SAME', reuse=None, name='conv2',use_bias = False)
conv3 = tf.layers.conv2d(conv2, 16, [3, 3], padding='SAME', reuse=True, name='conv2',use_bias = False)
print('conv1', conv1) # conv/BiasAdd:0 运算的名称
print('conv2', conv2) # conv/BiasAdd:0
print('conv3', conv3) # (10,32,32,16)

layer4 = tf.reduce_mean(conv3, [1,2], keepdims = True) # (10,1,1,16) global average pooling
layer5 = tf.reshape(layer4, [-1, 16]) # (10,16)
layer6 = tf.layers.dense(layer5, 20)  # (10,20)
print('layer4', layer4)
print('layer5', layer5)
print('layer6', layer6)

ohe = tf.one_hot(labels,20,dtype = tf.int32)
loss = tf.losses.softmax_cross_entropy(ohe, layer6)
grad1 = tf.gradients(loss, [conv2])[0] # 对 grad1 的梯度
grad2 = tf.gradients(loss, [conv3])[0] # 对 grad2 的梯度
# print('conv1', conv1) # conv/BiasAdd:0 运算的名称
# print('conv2', conv2) # conv/BiasAdd:0
# print('conv3', conv3)
# print([x.name for x in tf.global_variables()]) # 卷积层的名称
# ['conv1/kernel:0', 'conv2/kernel:0', 'dense/kernel:0', 'dense/bias:0']

train_op = tf.train.MomentumOptimizer(0.1, 0.9, use_nesterov = True).minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
     sess.run(init)
     for i in range(2):
         print('===============')
         result,_ = sess.run([layer6, train_op], feed_dict = {inputs:x})
         grad1_res = sess.run(grad1,feed_dict = {inputs:x})
         grad2_res = sess.run(grad2,feed_dict = {inputs:x})
         
'''
print([x.name for x in tf.global_variables()]) # 卷积层的名称
# [u'conv/kernel:0', u'conv/bias:0'] # share both weights and bias
print('conv1 name', conv1.name) # conv/BiasAdd:0 运算的名称
print('conv2 name', conv2.name) # conv/BiasAdd:0
print('conv3 name', conv3.name)

# print('conv1', conv1)
# print('conv2', conv2)
# conv1 name conv/BiasAdd:0
# conv2 name conv_1/BiasAdd:0
init = tf.global_variables_initializer()
with tf.Session() as sess:
     sess.run(init)
     result = sess.run(output)
'''

'''
# Note: tf.get_variable() 先定义变量
# 然后用 tf.nn.conv2d 定义卷积操作
import tensorflow as tf
# 不同卷积层之间的权值共享
batch_size = 100
inputs = tf.placeholder(tf.float32, (batch_size, 64, 64, 3), name='inputs')
weights = tf.get_variable(name='weights', shape=[5, 5, 3, 16], dtype=tf.float32)
w_1 = tf.get_variable(name='w_1', shape=[5, 5, 16, 16], dtype=tf.float32)
# tf.nn.conv2d(inputs, filters)
# filters: [filter_height, filter_width, in_channels, out_channels]

with tf.variable_scope("conv1"):
    hidden_layer_1 = tf.nn.conv2d(input=inputs, filter=weights, strides=[1, 1, 1, 1], padding="SAME")
    print('hidden_layer_1',hidden_layer_1) # (100,64,64,16)
    # 输出结果不同，是不同输出层，但是用到的卷积层函数是相同的
with tf.variable_scope("conv2", reuse=None):
    hidden_layer_2 = tf.nn.conv2d(input=hidden_layer_1, filter=w_1,strides=[1, 1, 1, 1], padding="SAME")
    print('hidden_layer_2', hidden_layer_2) # (100, 64, 64, 16)
    
with tf.variable_scope("conv2", reuse=True):
    hidden_layer_3 = tf.nn.conv2d(input=hidden_layer_2, filter=w_1,strides=[1, 1, 1, 1], padding="SAME")
    print('hidden_layer_3', hidden_layer_3) # (100, 64, 64, 16)
    
print([x.name for x in tf.global_variables()]) # 卷积层的名称
# ['weights:0','w_1:0']

# Question: conv2和conv2_1不同
# hidden_layer_1 Tensor("conv1/Conv2D:0", shape=(100, 64, 64, 16), dtype=float32)
# hidden_layer_2 Tensor("conv2/Conv2D:0", shape=(100, 64, 64, 16), dtype=float32)
# hidden_layer_3 Tensor("conv2_1/Conv2D:0", shape=(100, 64, 64, 16), dtype=float32)
'''
                                  
