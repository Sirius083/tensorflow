# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 15:39:01 2019

@author: Sirius

首先生成 batch_size=1的pb文件，再进行params和flops
需要更改以下内容
1. import 的 model
2. 定义 model
3. output_node_name
4. model_dir
5. input tensor的大小
6. min-error 文件夹中需要有stats.json
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"     


import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

model_dir = r'E:\chromeDownloads\SparseNet-master\src\train_log\sparse-40-24-c10-single-fisrt150-second225-max300'
output_node_name = 'output' 

# =============== params & flops ================
from tensorflow.python.framework import graph_util
from tensorpack import TowerContext 

def load_pb(pb):
    with tf.gfile.GFile(pb, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph

g = tf.Graph()
sess = tf.Session(graph=g)

# 定义model
from models.densecat import Model
model = Model(depth = 40, growth_rate=24, fetch="sparse", bottleneck=False, compression=1, dropout=0, num_classes=10)


with g.as_default():
     input_tensor = [tf.constant(np.zeros([1, 32, 32, 3]), dtype=tf.float32), tf.constant(np.zeros([1]), dtype=tf.int32)]
     with TowerContext('', is_training=False):
          model._build_graph(input_tensor)
     sess.run(tf.global_variables_initializer())
     params = tf.profiler.profile(g, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())

output_graph_def = graph_util.convert_variables_to_constants(sess, g.as_graph_def(), [output_node_name])

with tf.gfile.GFile('graph.pb', "wb") as f:
     f.write(output_graph_def.SerializeToString())

# load frozen graph
g2 = load_pb('./graph.pb')
with g2.as_default():
     flops = tf.profiler.profile(g2, options = tf.profiler.ProfileOptionBuilder.float_operation())
     print('*** FLOP after freezing /G')
     print(flops.total_float_ops/1e9)
     print('*** total params /M')
     print(params.total_parameters/1e6)

# ======================= model_validation_error ==================
import json
json_path = os.path.join(model_dir, 'stats.json')
with open(json_path) as f:
    data = json.load(f)
val_error = [i['validation_error'] for i in data]
print('min validation error {0:.0%}'.format(np.min(val_error)))

