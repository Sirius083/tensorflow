# ================================ version 1
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


# ==================================== version 2
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

      
