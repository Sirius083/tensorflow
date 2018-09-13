# Question: https://stackoverflow.com/questions/42862300/tensorflow-reuse-variable-with-tf-layers-conv2d
# variable get reused in the two convo  
# two layers share the name but not share the computation  
import tensorflow as tf

x = tf.random_normal(shape=[10, 32, 32, 3])

# Note: reuse=tf.AUTO_REUSE, first time False, second time True
conv1 = tf.layers.conv2d(x, 16, [3, 3], padding='SAME', reuse=None, name='conv')
conv2 = tf.layers.conv2d(x, 16, [3, 3], padding='SAME', reuse=True, name='conv')

print([x.name for x in tf.global_variables()])
print('conv1 name', conv1.name) # conv/BiasAdd:0
print('conv2 name', conv2.name) # conv/BiasAdd:0
