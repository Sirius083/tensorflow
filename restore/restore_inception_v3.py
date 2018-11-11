# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 16:38:30 2018

@author: Sirius

用tensorflow官网2016年pre-trained模型
无法得到正常结果
"""
'''
import tensorflow as tf
import os
import numpy as np

model_dir = r'E:\TF\TF_official_models\research\slim\inception_v3_pretrained\inception_2015'              
pb_name = 'classify_image_graph_def.pb'
pb_path = os.path.join(model_dir, pb_name)

with tf.Graph().as_default() as g:

     with tf.gfile.FastGFile(pb_path,'rb') as file:
          graph_def = tf.GraphDef()
          graph_def.ParseFromString(file.read())
          tf.import_graph_def(graph_def,name='')

tensor_name_softmax = "softmax:0"
tensor_name_softmax_logits = "softmax/logits:0"
tensor_name_transfer_layer = "pool_3:0"
logits = graph.get_tensor_by_name(tensor_name_softmax) # (1,1008)
unscaled_logits = graph.get_tensor_by_name(tensor_name_softmax_logits) # (1,1008)
transfer = graph.get_tensor_by_name(tensor_name_transfer_layer) # (1,1,1,2048)

'''
# restore all variables from a ckpt file
import cv2
import numpy as np    
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets

ckpt_path = 'E:/TF/TF_official_models/research/slim/inception_v3_pretrained/inception_v3.ckpt'

with tf.Graph().as_default() as g:
	image_tensor = tf.placeholder(tf.float32, shape=(None, 299, 299, 3))
	with slim.arg_scope(slim.nets.inception.inception_v3_arg_scope()):
	     logits, endpoints = slim.nets.inception.inception_v3(image_tensor,
	                                                         num_classes=1001,
	                                                         is_training=False)   
	     saver = tf.train.Saver()

# with tf.Session(graph = g) as sess:
sess = tf.Session(graph = g)
saver.restore(sess,ckpt_path)

# tensor_name_logits = endpoints['Predictions']
inputs = g.get_tensor_by_name("Const:0") # (1,1001)
logits = g.get_tensor_by_name('InceptionV3/Predictions/Reshape_1:0') # (1,1001)

#==============================================================================
# classify image
import os

test_dir = r'E:\TF\TF_official_models\research\slim\inception_v3_pretrained\test_pictures'
imgnumpy = np.ndarray((2,299,299,3))

name = ['panda.jpg', 'panda_adv.jpg']
pic = os.path.join(test_dir, name[0])
pic_adv = os.path.join(test_dir, name[1])


img = cv2.imread(pic) # (100,100,3)
'''
img = cv2.resize(img,(299, 299))/2550.0
imgnumpy[0] = img

img_adv = cv2.imread(pic_adv)
img_adv = cv2.resize(img,(299, 299))/255.0
imgnumpy[1] = img_adv


output = sess.run(logits, feed_dict={image_tensor: imgnumpy})
print('output shape', output.shape) # (2,1001)
'''
pred = sess.run(logits, feed_dict={image_tensor: tmp})


#==============================================================================
# inception-v3 中 evaludation的方法
central_fraction = 0.875
height = 299
width = 299
image = tf.constant(img)
image = tf.image.convert_image_dtype(image, dtype=tf.float32)
image = tf.image.central_crop(image, central_fraction=0.875)
image = tf.expand_dims(image, 0)
image = tf.image.resize_bilinear(image, [height, width],align_corners=False)
# image = tf.squeeze(image, [0])
image = tf.subtract(image, 0.5)
image = tf.multiply(image, 2.0)
sess_1 = tf.Session()
tmp = sess_1.run(image)
pred = sess.run(logits, feed_dict={image_tensor: tmp})[0]
#==============================================================================
# 找到名称编号和class的对应号
from Imagenet_NameLookup import *
name_lookup = NameLookup()
k = 10 # only show first 10 predicted results
top_k = pred.argsort()[::-1][:k] # np.argsort: 从小到大的 index
# result = [name_lookup.cls_to_name(c, only_first_name = True) for c in top_k]

for i in range(len(top_k)):
    c = top_k[i]
    name = name_lookup.cls_to_name(cls = c, only_first_name = True)
    score = pred[c]
    print("{0:>6.2%}:{1}".format(score,name))
    
