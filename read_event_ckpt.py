# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 14:44:17 2018

@author: dell4mc
"""
# Note: 可以从tensorboard中直接下载csv文件
# 从checkpoint的ckpt文件中读取变量的值
from tensorflow.python import pywrap_tensorflow
import os
checkpoint_path = os.path.join(model_dir, "model.ckpt")
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor_name: ", key)
    print(reader.get_tensor(key)) # Remove this is you want to print only variable names


# 从events文件中回复变量的值
import tensorflow as tf
train_accuracy_1 = []
event_dir = 'E:\\resnet\\models-master\\official\\resnet\\imagenet_train_18\\events.out.tfevents.1531975444.DESKTOP-L32SK0R'
for event in tf.train.summary_iterator(event_dir):
    for value in event.summary.value:
        if value.tag == 'train_accuracy_1':
            train_accuracy_1.append(value.simple_value)
          
'''
# 从checkpoint的ckpt文件中读取变量的值
from tensorflow.python import pywrap_tensorflow
import os
checkpoint_path = os.path.join(model_dir, "model.ckpt")
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor_name: ", key)
    print(reader.get_tensor(key)) # Remove this is you want to print only variable names


# 从events文件中回复变量的值
for event in tf.train.summary_iterator('F:\\tmp3\\cifar10_model\\events.out.tfevents.1526303210.DESKTOP-G76IDEK'):
    for value in event.summary.value:
        if value.tag == 'l2_loss':
            lossList.append(value.simple_value)

'''
