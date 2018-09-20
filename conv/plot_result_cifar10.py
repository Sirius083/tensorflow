# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 15:42:51 2018

@author: Sirius

cifar10 结果比较
"""

import tensorflow as tf

original_train = []
original_vali = []
original_train_error = []
original_vali_error = []

event_dir = r'E:\TF\ShareWeight\cifar10\train_dir\resnet-original-n-218\events.out.tfevents.1537109753.DESKTOP-L32SK0R'
for event in tf.train.summary_iterator(event_dir):
    for value in event.summary.value:
        if value.tag == 'val_loss':
            original_train.append(value.simple_value)
        if value.tag == 'train_loss':
            original_vali.append(value.simple_value)
        if value.tag == 'val_top1_error':
            original_vali_error.append(value.simple_value)
        if value.tag == 'train_top1_error':
            original_train_error.append(value.simple_value)


share_train = []
share_vali = []
share_train_error = []
share_vali_error = []

event_dir = r'E:\TF\ShareWeight\cifar10\train_dir\resnet-share-n-218\events.out.tfevents.1537000527.DESKTOP-L32SK0R'
for event in tf.train.summary_iterator(event_dir):
    for value in event.summary.value:
        if value.tag == 'val_loss':
            share_train.append(value.simple_value)
        if value.tag == 'train_loss':
            share_vali.append(value.simple_value)
        if value.tag == 'val_top1_error':
            share_vali_error.append(value.simple_value)
        if value.tag == 'train_top1_error':
            share_train_error.append(value.simple_value)


import matplotlib.pyplot as plt                   
fig = plt.figure(figsize = (24,10))
plt.subplot(1,2,1)
plt.plot(original_train, label = 'original 218 train loss', linestyle='--',c='r')
plt.plot(original_vali, label = 'original 218 test loss',c='r')
plt.plot(share_train, label = 'share 218 train loss', linestyle='--',c='b')
plt.plot(share_vali, label = 'share 218 test loss',c='b')
plt.legend()
plt.title('cifar10 218 layer loss')


plt.subplot(1,2,2)
plt.plot(original_train_error, label = 'original 218 train error', linestyle='--',c='r')
plt.plot(original_vali_error, label = 'original 218 test error',c='r')
plt.plot(share_train_error, label = 'share 218 train error', linestyle='--',c='b')
plt.plot(share_vali_error, label = 'share 218 test error',c='b')
plt.legend()
plt.title('cifar10 218 layer error')
fig.savefig('cifar10-218.png',dpi=fig.dpi)
           
