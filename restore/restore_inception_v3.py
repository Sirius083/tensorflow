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

with tf.Session(graph = g) as sess:
    saver.restore(sess,ckpt_path)

tensor_name_logits = 'InceptionV3/Predictions/Reshape_1:0'
logits = g.get_tensor_by_name(tensor_name_logits) # (1,1001)
