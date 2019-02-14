# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 11:26:42 2019

@author: Sirius
"""

# 输入 imagenet 验证集的图片，输出在 panda_389中对应的cls标号

def predict(image_path, graph_pb_path):
    import tensorflow as tf
    tf.reset_default_graph()
    
    import os
    current_path = os.getcwd()
    
    # change directory to import module
    os.chdir(r'E:\resnet_test')
    
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0) # only difference
    
    # ====================================
    #        preprocess image
    # ====================================
    import numpy as np
    # from tensorflow.python.platform import gfile    
    # official preprocess
    
    # 单独预测一张图片时需要
    # ***************************************************
    # from imagenet_preprocessing import preprocess_image
    # ***************************************************
    
    image_buffer = tf.gfile.FastGFile(image_path,'rb').read()
    image = preprocess_image(image_buffer,bbox = None, output_height = 224, 
                             output_width = 224, num_channels = 3, is_training = False)
    image = tf.expand_dims(image, 0)
    with tf.Session() as sess:
        img = sess.run(image)
        
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
        
    prob = softmax(np.squeeze(res))
    top10_id = prob.argsort()[-10:][::-1] # top10 class id 
    top10_prob = prob[top10_id]           # top10 class probability
    
    os.chdir(current_path) # change to original directory 
    
    return top10_id, top10_prob

GRAPH_PB_PATH = r'E:\resnet_test\export_dir\resnet_50\1547058655'
image_path = r'E:\resnet_test\test_pictures\panda.jpg'
top10_id, top10_prob = predict(image_path)

