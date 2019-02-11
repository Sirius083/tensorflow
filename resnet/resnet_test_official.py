# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 19:23:48 2019

@author: Sirius
"""

# 通过export_dir对图片进行预测
# original resnet code
# tensorflow use panda as 389

# .pb: MetaGraphDef hold graph structure
# variables: learned weights
import tensorflow as tf
tf.reset_default_graph()


import os
os.chdir(r'E:\resnet_test')

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference
# ====================================
#         class_to_name
# ====================================
# imagenet_class_name.txt
name_path = r'E:\resnet_test\imagenet1000_clsidx_to_labels.txt'
with open(name_path, 'r') as f:
    content = f.readlines()

class_to_name = {}
for x in content:
    k,v = x.rstrip().split(':')
    class_to_name[int(k)+1] = v[2:-2]
    
# ====================================
#        preprocess image
# ====================================
import numpy as np
# from tensorflow.python.platform import gfile
GRAPH_PB_PATH = r'E:\resnet_test\export_dir\resnet_50\1547058655'
image_path = r'E:\resnet_test\test_pictures\chicken.jpg'


# ===================================
#         image process
# ===================================
'''
import cv2
img = cv2.imread(image_path)
img = cv2.resize(img, (224, 224))
img = np.expand_dims(img, axis = 0)
'''

# official preprocess
from imagenet_preprocessing import *
image_buffer = tf.gfile.FastGFile(image_path,'rb').read()
image = preprocess_image(image_buffer,bbox = None, output_height = 224, 
                         output_width = 224, num_channels = 3, is_training = False)
image = tf.expand_dims(image, 0)

with tf.Session() as sess:
    img = sess.run(image)
    
import matplotlib.pyplot as plt
plt.imshow(img[0,...])

# ====================================
#       load pretrained model
# ====================================
with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess,["serve"], GRAPH_PB_PATH)
    # resnet_restore
    # res = sess.run('resnet_model/final_dense:0',feed_dict = {'input_tensor_change_name:0':img}) # (128,1001)
    graph = tf.get_default_graph()
    inputs = graph.get_tensor_by_name('input_tensor:0')
    model = graph.get_tensor_by_name('resnet_model/final_dense:0')
    # res = sess.run('resnet_model/final_dense:0',feed_dict = {'input_tensor:0':img}) # (1,1001)
    res = sess.run(model, {inputs:img})

# print final result
def print_score(res,class_to_name):
    prob = softmax(np.squeeze(res))
    top10_id = prob.argsort()[-10:][::-1]
    top10 = prob[top10_id]
    
    # 按照百分号输出浮点数，保留两位小数
    # print('{0:>6.2%}'.format(9.5639473e-01))
    for ind in top10_id:
        print("{:d} {:>6.2%} : {}".format(ind, prob[ind], class_to_name[ind])) 

print_score(res,class_to_name)


