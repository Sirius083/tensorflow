'''
共享conv2d的方法
1. 在定义模型前定义top_scope, 后面索引变量通过top_scope进行索引
2. 在定义模型迁定义g = tf.get_default_graph(), 通过g.get_tensor_by_name() 进行索引
3. 通过tf.variable_scope() 在一个变量域中进行共享

其中最简单的是通过tf.variable_scope进行实现
在同一个variable_scope, 只用在该scope下的名称就可以进行变量共享
不同的variable_scope不能共享变量
'''

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
import tensorflow as tf

x = tf.zeros([10,32,32,64], tf.float32)
with tf.variable_scope('conv'):
    x = tf.layers.conv2d(x, 64, 3, padding = 'SAME', use_bias = False, name = 'conv1')
    x = tf.layers.conv2d(x, 64, 3, padding = 'SAME', use_bias = False, name = 'conv1', reuse = True)
    x = tf.layers.conv2d(x, 64, 3, padding = 'SAME', use_bias = False, name = 'conv2')
    x = tf.layers.conv2d(x, 64, 3, padding = 'SAME', use_bias = False, name = 'conv2', reuse = True)
with tf.variable_scope('layer'):
    x = tf.layers.conv2d(x, 64, 3, padding = 'SAME', use_bias = False, name = 'conv1')
    x = tf.layers.conv2d(x, 64, 3, padding = 'SAME', use_bias = False, name = 'conv1', reuse = True)  
# print('x', x)
allvars = tf.global_variables()
for var in allvars:
    print(var)

# 创建 test_graph
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
test_graph_path = r'C:\Users\Sirius\Desktop\resnet\test_identity'
summary_writer = tf.summary.FileWriter(test_graph_path, sess.graph)
sess.close()
