# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 12:33:34 2018

@author: Sirius
"""
# 为什么要指定输出大小 
# [1,5,5,3] 和 [1,6,6,3] 的图经过卷积得到了相同的大小 [1,3,3,1]
import tensorflow as tf
 
x1 = tf.constant(1.0, shape=[1,3,3,1])

# x2 和 x3两种输入都可以得到x1的conv结果，但是
x2 = tf.constant(1.0, shape=[1,6,6,3])
x3 = tf.constant(1.0, shape=[1,5,5,3])
 
kernel = tf.constant(1.0, shape=[3,3,3,1])

# 几个参数
# value: [batch, height, width, in_channels]
# filter: [height, width, output_channels, in_channels]
# output_shape: filter's in_channels dimensions
# strides: strides of input tensor
# padding: default "SAME" 
# data_format: default "NHWC"
# name 
 
x1_decov = tf.nn.conv2d_transpose(x1,kernel,output_shape=[1,6,6,3],
    strides=[1,2,2,1],padding="SAME")

x1_decov2 = tf.nn.conv2d_transpose(x1,kernel,output_shape=[1,5,5,3],
    strides=[1,2,2,1],padding="SAME")

x2_cov = tf.nn.conv2d(x2, kernel, strides=[1,2,2,1], padding="SAME")
x3_cov = tf.nn.conv2d(x3, kernel, strides=[1,2,2,1], padding="SAME")

sess = tf.Session()
tf.global_variables_initializer().run(session=sess)
print(sess.run(x1_decov).shape)
print(sess.run(x1_decov2).shape)
print(sess.run(x2_cov).shape)
print(sess.run(x3_cov).shape)
