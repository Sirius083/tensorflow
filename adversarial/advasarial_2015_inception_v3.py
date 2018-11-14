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
import pandas as pd
import inception
from inception import NameLookup
import tensorflow as tf

import matplotlib.pyplot as plt

# 给 advasarial example 带入具体数值
# session = tf.Session(graph=model.graph)

# 函数参数
images_dir = r'E:\TensorFlow-Tutorials-master\test_pictures'
image_path =os.path.join(images_dir, 'suanpan.jpg')
cls_target = 300
max_iterations=100
required_score=0.99
noise_limit=3.0

NUM_CLASSES = 1008
k = 10          # 输出结果显示前10类
Y_TARGET = 300  # 要转化到的目标类
eps = 0.1
k = 5          # 输出结果显示前10类
Y_TARGET = 300  # 要转化到的目标类
epochs = 100


def step_fgsm(x, eps, logits):
    label = tf.argmax(logits,1)
    one_hot_label = tf.one_hot(label, NUM_CLASSES)
    cross_entropy = tf.losses.softmax_cross_entropy(one_hot_label,
                                                    logits,
                                                    label_smoothing=0.1)
    x_adv = x + eps*tf.sign(tf.gradients(cross_entropy,x)[0])
    x_adv = tf.clip_by_value(x_adv,0.0,255.0)
    return tf.stop_gradient(x_adv) # 问题：这里为什么要返回step_gradients


'''
def step_targeted_attack(x, eps, target_class, logits):
    # tf.gradient should receive the inception model input tensor, not the image.
    cross_entropy = tf.losses.softmax_cross_entropy(target_class,
                                                  logits,
                                                  label_smoothing=0.1)

    x_adv = x - eps * tf.sign(tf.gradients(cross_entropy, x)[0])
    x_adv = tf.clip_by_value(x_adv, 0.0, 1.0)
    return tf.stop_gradient(x_adv)
'''

model = inception.Inception()
y_pred = model.y_pred
y_logits = model.y_logits
namelookup = NameLookup()

def get_scores(pred):
    # 输入预测的softmax, 返回对应类名称和得分
    namelookup = NameLookup() # 为什么会有1008类
    idx = pred.argsort()
    top_k = idx[-k:]
    name = namelookup.cls_to_name(cls=top_k[-1], only_first_name=True)
    score = pred[top_k[-1]]
    return name, score

def get_scores_cls(name_sorce, pred):
    # 输入：结果softmax
    # 返回：指定类别的得分（正确标签的得分）
    namelookup = NameLookup()
    cls_ = namelookup.name_to_cls(name_sorce, only_first_name=True)
    score = pred[cls_]
    return score


def print_scores(pred):
    namelookup = NameLookup() # 为什么会有1008类
    idx = pred.argsort()
    top_k = idx[-k:]
    for cls_ in reversed(top_k):
        name = namelookup.cls_to_name(cls=cls_, only_first_name=True)
        score = pred[cls_]
        print("{0:>6.2%} : {1}".format(score, name))



def write_result(pred, result):
    # 将预测结果写在 excel文件中
    # result = pd.DataFrame(columns=['1','2','3','4','5'])
    namelookup = NameLookup()
    idx = pred.argsort()
    top_k = idx[-k:]
    scorelist = []
    namelist = []
    
    for cls_ in reversed(top_k):
        name = namelookup.cls_to_name(cls=cls_, only_first_name=True)
        score = round(pred[cls_] * 100,2)
        namelist = namelist + [name]
        scorelist = scorelist + [score]
    
    n = len(result)
    result.loc[n+2] =  namelist # 前两行：初始预测结果
    result.loc[n+3] =  scorelist
    return result
    
        
        

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
    # target_class = tf.reshape(tf.one_hot(Y_TARGET,NUM_CLASSES),[1,NUM_CLASSES])
    softmax_tensor = model.graph.get_tensor_by_name("softmax:0")
    adv_image_tensor = step_fgsm(image_tensor, eps, softmax_tensor)
    # adv_image_tensor, noise = step_targeted_attack(image_tensor, eps, target_class, softmax_tensor)
    # adv_image_tensor = step_targeted_attack(image_tensor, eps, target_class, softmax_tensor)
    # adv_noise = np.zeros((1,299,299,3))
    adv_image = image
    result = pd.DataFrame(columns=['rank_1','rank_2','rank_3','rank_4','rank_5']) # 分类结果在前五类的概率
    result = write_result(predictions, result)
    for i in range(epochs):
        adv_image = sess.run(adv_image_tensor, {tensor_name_resized_image:adv_image})
        noise = image - adv_image
        if i % 10 == 0:
            print("Iteration" + str(i))
            predictions_adv = sess.run(y_pred, {image_tensor: adv_image})
            predictions_adv = np.squeeze(predictions_adv)
            print_scores(predictions_adv)
            print('\n')
            result = write_result(predictions_adv, result)
    
    # 将计算结果保存在txt文件中
    idx = [list(np.repeat(i, 2)) for i in np.arange(0,100,10)]
    idx = [j for i in idx for j in i]
    result_index = ['original', 'original'] + ['itr_' + str(i)  for i in idx]
    result.index = result_index
    
    '''
    # DF TO EXCEL
    from pandas import ExcelWriter
    writer = ExcelWriter('macaw_fgsm.xlsx')
    result.to_excel(writer)
    writer.save()
    '''
    # print('adv_image shape', adv_image.shape)
    # print('a shape', a.shape)
    
    # 找到原始图像和对抗样本的分数
    # ================================================
    # plot images
    name_source, score_source_org = get_scores(predictions)
    name_adv, score_adv = get_scores(predictions_adv)
    score_adv_orginal_cls = get_scores_cls(name_source, predictions_adv)
    
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,3,figsize = (10,10))
    fig.subplots_adjust(hspace = 0.1, wspace = 0.1) # adjust vertical spacing
    
    # plot images, note that the pixel-values are normalized to the [0.0, 0,1]
    ax = axes.flat[0]
    ax.imshow(image[0]/255.0)
    msg = "Original Image:\n{0} ({1:.2%})"
    xlabel = msg.format(name_source, score_source_org)
    ax.set_xlabel(xlabel)
    # ax.set_title('original macaw')

    # Plot the noisy image.
    ax = axes.flat[1]
    ax.imshow(adv_image[0]/ 255.0)
    msg = "Image + Noise:\n{0} ({1:.2%})\n{2} ({3:.2%})"
    xlabel = msg.format(name_source, score_adv_orginal_cls, name_adv, score_adv)
    ax.set_xlabel(xlabel)
    # ax.set_title('adversarial macaw fgsm')

    # Plot the noisy image.
    ax = axes.flat[2]
    ax.imshow(noise[0])
    xlabel = "noise macaw fgsm"
    ax.set_xlabel(xlabel)
    # ax.set_title('noise macaw fgsm')


    # remove ticks from all the plots
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    
    # 将图像保存在png文件中
    fig.savefig('test_pictures/suanpan_fgsm_pic.png')
    plt.show()
    
    

    





