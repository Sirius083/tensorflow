# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 16:32:46 2018

@author: Sirius


tf.nn.max_pool_with_argmax:
performs max pooling on the input and outputs both max values and indices

再对value和index进行处理

requirement:
1: > tensorflow 1.0
2: GPU use only

tf.reset_default_graph()
sess.close()
"""

import tensorflow as tf

# implementation from
# https://github.com/rayanelleuch/tensorflow/blob/b46d50583d8f4893f1b1d629d0ac9cb2cff580af/tensorflow/contrib/layers/python/layers/layers.py#L2291-L2327
    
def unpool_2d(pool, 
              ind, 
              stride=[1, 2, 2, 1], 
              scope='unpool_2d'):
  """Adds a 2D unpooling op.
  https://arxiv.org/abs/1505.04366
  Unpooling layer after max_pool_with_argmax.
       Args:
           pool:        max pooled output tensor
           ind:         argmax indices
           stride:      stride is the same as for the pool
       Return:
           unpool:    unpooling tensor
  """
  with tf.variable_scope(scope):
    input_shape = tf.shape(pool)
    output_shape = [input_shape[0], input_shape[1] * stride[1], input_shape[2] * stride[2], input_shape[3]]

    flat_input_size = tf.reduce_prod(input_shape)
    flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

    pool_ = tf.reshape(pool, [flat_input_size])
    batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64), dtype=ind.dtype), 
                                      shape=[input_shape[0], 1, 1, 1])
    b = tf.ones_like(ind) * batch_range
    b1 = tf.reshape(b, [flat_input_size, 1])
    ind_ = tf.reshape(ind, [flat_input_size, 1])
    ind_ = tf.concat([b1, ind_], 1)

    ret = tf.scatter_nd(ind_, pool_, shape=tf.cast(flat_output_shape, tf.int64))
    ret = tf.reshape(ret, output_shape)

    set_input_shape = pool.get_shape()
    set_output_shape = [set_input_shape[0], set_input_shape[1] * stride[1], set_input_shape[2] * stride[2], set_input_shape[3]]
    ret.set_shape(set_output_shape)
    return ret


# image_tensor = tf.expand_dims(tf.Variable(image, dtype=tf.float32), 0)
image_tensor = tf.random_normal([1,4,4,1])
# output_maxpool = tf.nn.max_pool(image_tensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
output, argmax = tf.nn.max_pool_with_argmax(image_tensor, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool2')

unpool_result = unpool_2d(output, argmax, stride=(1,2,2,1), scope='unpool')

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
# 必须在一个session中进行，否则结果不正确
image_tensor_run,output_run,argmax_run,unpool_result_run = \
sess.run([image_tensor,output,argmax,unpool_result])
image_tensor_run = image_tensor_run[0,:,:,0]
output_run = output_run[0,:,:,0]
argmax_run = argmax_run[0,:,:,0]
unpool_result_run = unpool_result_run[0,:,:,0]
sess.close()
