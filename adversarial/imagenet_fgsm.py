# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 16:26:45 2018

@author: Sirius
参考
https://cv-tricks.com/how-to/breaking-deep-learning-with-adversarial-examples-using-tensorflow/

使用inception v3 进行图像分类
https://www.jianshu.com/p/cc830a6ed54b

imagenet 1001个类别：第零个类别是background
         1008个类别：old_syntax.txt,未找到
         
tf.reset_default_graph()
"""
import os
import numpy as np
import inception
from inception import NameLookup
import tensorflow as tf


# 函数参数
images_dir = r'E:\TensorFlow-Tutorials-master\test_pictures'
image_path =os.path.join(images_dir, 'macaw.jpg')
cls_target = 300
max_iterations=100
required_score=0.99
noise_limit=3.0
NUM_CLASSES = 1008
k = 10          # 输出结果显示前10类
Y_TARGET = 300  # 要转化到的目标类
eps = 0.01
model = inception.Inception()
y_pred = model.y_pred
y_logits = model.y_logits
namelookup = NameLookup()

def step_fgsm(x, eps, logits):
    label = tf.argmax(logits,1)
    one_hot_label = tf.one_hot(label, NUM_CLASSES)
    cross_entropy = tf.losses.softmax_cross_entropy(one_hot_label,
                                                    logits,
                                                    label_smoothing=0.1)
    x_adv = x + eps*tf.sign(tf.gradients(cross_entropy,x)[0])
    x_adv = tf.clip_by_value(x_adv,0.0,255.0)
    return tf.stop_gradient(x_adv) # 问题：这里为什么要返回step_gradients


def print_scores(pred):
    namelookup = NameLookup() # 为什么会有1008类
    idx = pred.argsort()
    top_k = idx[-k:]
    for cls_ in reversed(top_k):
        name = namelookup.cls_to_name(cls=cls_, only_first_name=True)
        score = pred[cls_]
        print("{0:>6.2%} : {1}".format(score, name))

tensor_name_input_jpeg = "DecodeJpeg/contents:0"
tensor_name_resized_image = "ResizeBilinear:0"
sess = tf.Session(graph = model.graph)

with model.graph.as_default():
    image_tensor = model.graph.get_tensor_by_name(tensor_name_resized_image) # "ResizeBilinear:0"
    image_raw_data = tf.gfile.FastGFile(image_path,'rb').read()
    image = sess.run(image_tensor, {tensor_name_input_jpeg: image_raw_data}) # 将输入归一化到[0,1]区间
    predictions = sess.run(y_pred, {image_tensor: image})
    predictions = np.squeeze(predictions)
    print_scores(predictions)
    
    print('Generating adversarial examples...\n')
    softmax_tensor = model.graph.get_tensor_by_name("softmax:0")
    adv_image_tensor = step_fgsm(image_tensor, eps, softmax_tensor)
    adv_image = image
    for i in range(epochs):
        adv_image = sess.run(adv_image_tensor, {tensor_name_resized_image:adv_image})
        if epochs % 10 == 0:
            print("Iteration" + str(i))
            predictions = sess.run(y_pred, {image_tensor: adv_image})
            predictions = np.squeeze(predictions)
            print_scores(predictions)
            print('\n')

