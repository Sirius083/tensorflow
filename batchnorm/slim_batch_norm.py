# 在slim中间batch_norm在activation后面定义，因此可以直接在arg_scope中添加
# slim 中定义conv2d的原代码: 
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/layers.py
# 在slim中使用batch norm的用法
# https://github.com/tensorflow/tensorflow/issues/5663
# 函数定义 relu --> batch norm
# 这里的比较需要在两个console中进行，否则无法比较
import tensorflow as tf
import tensorflow.contrib.slim as slim

data = tf.random_normal(shape=(64, 16, 16, 3), seed=10)
winit =tf.contrib.layers.xavier_initializer(seed=1)

# version1
relu1 = slim.conv2d(data, 16, (3,3), normalizer_fn=slim.batch_norm,
                    scope='conv_in',
                    weights_initializer = winit)

# version2

conv_no_bn = slim.conv2d(data, 16, (3,3), scope='conv_out', activation_fn=None,  
                         weights_initializer=tf.contrib.layers.xavier_initializer(seed=1))
alt_conv_bn = slim.batch_norm(conv_no_bn, is_training=True)
relu2 = tf.nn.relu(alt_conv_bn)



with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())
  relu_in = sess.run(relu1)
  relu_out = sess.run(relu2)
  print(sess.run(tf.equal(relu_in, relu_out)))
  
