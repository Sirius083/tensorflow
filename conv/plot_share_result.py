# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 15:59:43 2018

@author: Sirius
"""

# 比较share层不同n时的重复次数
import pandas as pd
import matplotlib.pyplot as plt   
original = r'E:\TF\ShareWeight\cifar10\train_dirresnet-original-n-20_error.csv'
share_1 = r'E:\TF\ShareWeight\cifar10\train_dirresnet-share-n-20_error.csv'
share_2 = r'E:\TF\ShareWeight\cifar10\train_dirresnet-share-n-32_error.csv'
share_3 = r'E:\TF\ShareWeight\cifar10\train_dirresnet-share-n-44_error.csv'
share_4 = r'E:\TF\ShareWeight\cifar10\train_dirresnet-share-n-56_error.csv'
share_5 = r'E:\TF\ShareWeight\cifar10\train_dirresnet-share-n-110_error.csv'

data = pd.read_csv(original)
data_1 = pd.read_csv(share_1)
data_2 = pd.read_csv(share_2)
data_3 = pd.read_csv(share_3)
data_4 = pd.read_csv(share_4)
data_5 = pd.read_csv(share_5)


fig = plt.figure(figsize = (12,10))
plt.plot(data['train_error'], label  = 'original 20 train error', linestyle='--',c='y',)
plt.plot(data_1['train_error'], label  = 'share 20 train error', linestyle='--',c='r',)
plt.plot(data_2['train_error'], label  = 'share 32 train error',linestyle ='--',c='g')
plt.plot(data_3['train_error'], label = 'share 44 train error', linestyle='--',c='b',alpha=0.7)
plt.plot(data_4['train_error'], label = 'share 56 train error', linestyle='--',c='brown')
plt.plot(data_5['train_error'], label = 'share 110 train error', linestyle='--',c='m')

plt.plot(data['validation_error'], label  = 'original 20 validation error', c='y',)
plt.plot(data_1['validation_error'], label  = 'share 20 validation error', c='r',)
plt.plot(data_2['validation_error'], label  = 'share 32 validation error',c='g')
plt.plot(data_3['validation_error'], label = 'share 44 validation error', c='b',alpha=0.7)
plt.plot(data_4['validation_error'], label = 'share 56 validation error', c='brown')
plt.plot(data_5['validation_error'], label = 'share 110 validation error',c='m')

plt.legend()
plt.title('cifar10 share top-1-error')


#========================================
# errors last step train
def moving_average(a, n=8) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

import numpy as np
array_0 = np.array(pd.read_csv(original)[-30:]['validation_error'])
array_1 = np.array(pd.read_csv(share_1)[-30:]['validation_error'])
array_2 = np.array(pd.read_csv(share_2)[-30:]['validation_error'])
array_3 = np.array(pd.read_csv(share_3)[-30:]['validation_error'])
array_4 = np.array(pd.read_csv(share_4)[-30:]['validation_error'])
array_5 = np.array(pd.read_csv(share_5)[-30:]['validation_error'])


array_0_ave = moving_average(array_0)
array_1_ave = moving_average(array_1 )
array_2_ave = moving_average(array_2)
array_3_ave = moving_average(array_3)
array_4_ave = moving_average(array_4)
array_5_ave = moving_average(array_5)


fig = plt.figure(figsize = (12,10))
plt.plot(array_0_ave, label  = 'original 20 validation error',  c='y')
plt.plot(array_1_ave, label  = 'share 20 validation error',  c='r')
# plt.plot(array_2_ave, label  = 'share 32 validation error',  c='g')
plt.plot(array_3_ave, label  = 'share 44 validation error',  c='b')
plt.plot(array_4_ave, label  = 'share 56 validation error',  c='brown', alpha = 0.7)
plt.plot(array_5_ave, label  = 'share 110 validation error',  c='m')
plt.legend()
plt.title('cifar100 share validation top-1-error last steps ')
