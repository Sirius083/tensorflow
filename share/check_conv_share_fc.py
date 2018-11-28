# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 10:53:13 2018

@author: Sirius

只要证明用reuse = True计算出来的梯度是正确的
在全连接层计算即可
tf.reset_default_graph()
sess.close()
"""
import numpy as np
import tensorflow as tf
g = tf.get_default_graph()
x = tf.random_normal(shape=[2, 8], seed = 7)
y = tf.constant([1,2])
label = tf.one_hot(y, depth = 10)
# 用到了两个全连接层，共享变量
fc1 = tf.layers.dense(x, 8, kernel_initializer = tf.contrib.layers.xavier_initializer(seed = 1), reuse = None, name = 'fc')
fc2 = tf.layers.dense(fc1, 8, reuse = True, name = 'fc')
logits = tf.layers.dense(fc2, 10, kernel_initializer = tf.contrib.layers.xavier_initializer(seed = 1))
entropy = tf.nn.softmax_cross_entropy_with_logits(labels = label, logits = logits)
loss = tf.reduce_mean(entropy, name = "loss")

# 获得图中tensor的名字
fc_kernel = g.get_tensor_by_name('fc/kernel:0')
fc_bias = g.get_tensor_by_name('fc/bias:0')

dense_kernel = g.get_tensor_by_name('dense/kernel:0')
dense_bias = g.get_tensor_by_name('dense/bias:0')

# 梯度计算
# sum(dy/dx) for y in ys
# logits: (2,10)
# fc2:    (2,8)
# kernel_dense: (8,10)
logits_grad = tf.gradients(loss, logits)
fc2_grad = tf.gradients(loss, fc2)
fc1_grad = tf.gradients(loss, fc1)
fc2_fc1_grad = tf.gradients(fc2, fc1)
tmp = tf.gradients(logits, fc2)

# =============================================================================
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(tf.global_variables())
# fc2_fc1_grad_eval = sess.run(fc2_fc1_grad)[0]

# print('fc2_fc1_grad_eval[0].shape',fc2_fc1_grad_eval[0].shape)
# print('fc_kernel.shape', fc_kernel.shape)


# lg = sess.run(logits_grad)
# f2 = sess.run(fc2_grad)
# tmp_eval = sess.run(tmp)
# dense_kernel_eval = sess.run(dense_kernel)

'''
# 16X16的梯度方程
import numpy as np
A = fc_kernel_eval
A_T = np.transpose(A)
upper = np.concatenate((A_T, np.zeros((8,8))), axis = 1)
lower = np.concatenate((np.zeros((8,8)), A_T), axis = 1)
whole = np.concatenate((upper,lower), axis = 0)
row = np.sum(A_T,axis = 0)
col = np.sum(A_T,axis = 1)
'''
x_eval, fc_kernel_eval, fc_bias_eval, fc1_eval, fc2_eval,fc2_fc1_grad_eval = \
sess.run([x, fc_kernel, fc_bias, fc1, fc2, fc2_fc1_grad])
fc1_tmp = np.matmul(x_eval, fc_kernel_eval) + fc_bias_eval
fc2_tmp = np.matmul(fc1_tmp, fc_kernel_eval) + fc_bias_eval

fc2_fc1_grad_eval = fc2_fc1_grad_eval[0]

