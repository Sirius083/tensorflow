# -*- coding: utf-8 -*-
"""
@author: Sirius

compare densenet-121 pretrained model label with true label
"""

import os
import numpy as np

synset_to_cls_path = r'E:\Data\imagenet_related\imagenet1000_synset_to_name.txt'
path_true_synset = r'E:\Data\imagenet_related\imagenet_2012_validation_synset_labels.txt'

# model predicted class index
predict = np.load('densenet_121_results.npy') # load
predict_label = np.argmax(predict, axis = 1)

# synset to cls
with open(synset_to_cls_path, 'r') as myfile:
     content = myfile.readlines()
     
content = [c.rstrip() for c in content]
cls_to_synset = {}

for i in range(len(content)):
    cls_to_synset[i] = content[i].split(' ')[0]
synset_to_cls = dict((v,k) for k,v in cls_to_synset.items())

# true label
with open(path_true_synset,'r') as f:
     content = f.readlines()
true_synset = [c.rstrip() for c in content]
true_label = [synset_to_cls[synset] for synset in true_synset]

correct_num = [predict_label[i] == true_label[i] for i in range(50000)]
print(sum(correct_num)/50000) # 72.91

