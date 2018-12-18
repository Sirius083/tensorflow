# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 09:42:44 2018

@author: Sirius

guided back propagation 定义

https://gist.github.com/falcondai/561d5eec7fed9ebf48751d124a77b087

tf.reset_default_graph()
"""

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

'''
# tensorflow 原始定义
@ops.RegisterGradient("Relu")
def _ReluGrad(op, grad):
  return gen_nn_ops.relu_grad(grad, op.outputs[0])
'''

'''
#=========== normal gradients ================
if __name__ == '__main__':
    # tf.reset_default_graph()
    with tf.Session() as sess:
        g = tf.get_default_graph()
        x = tf.constant([-10.])
        y = tf.nn.relu(x)
        # z = tf.reduce_sum(-y ** 2)
        z = tf.square(x)
        tf.initialize_all_variables().run()

        print(x.eval(), y.eval(), z.eval(), tf.gradients(z, x)[0].eval())
        # > [10.  2.] [10.  2.] -104.0 [-20.  -4.]
'''
        
#===========guided gradients================
@ops.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    return tf.where(0. < grad, gen_nn_ops.relu_grad(grad, op.outputs[0]), tf.zeros(grad.get_shape()))

if __name__ == '__main__':
    with tf.Session() as sess:
        g = tf.get_default_graph()
        x = tf.constant([10.])
        with g.gradient_override_map({'Relu': 'GuidedRelu'}):
            y = tf.nn.relu(x)
            # z = tf.reduce_sum(-y ** 2)
            # z = tf.multiply(x**2,-1)
            z = -y**2
        tf.global_variables_initializer()

        print(x.eval(), y.eval(), z.eval(), tf.gradients(z, x)[0].eval())
        # > [ 10.   2.] [ 10.   2.] -104.0 [ 0.  0.]

'''
#======================================================
#              naive implementation
#======================================================
# https://stackoverflow.com/questions/38340791/guided-back-propagation-in-tensorflow/38779798
# 在每个定义relu的地方都要定义
before_relu = f1(inputs, params)
after_relu = tf.nn.relu(before_relu)
loss = f2(after_relu, params, targets)

# derivative wrt after_relu
Dafter_relu = tf.gradients(loss, after_relu)[0]
# threshold gradients thay you send down
Dafter_relu_thresholded = tf.where(Dafter_relu < 0.0, 0.0, Dafter_relu)
# actual gradients wrt params
Dparams = tf.gradients(after_relu, params, grad_ys=Dafter_relu_thresholded)
'''
