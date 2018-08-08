# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 16:12:30 2018

@author: Sirius

iterate over batch to calculate whole dataset mean
tiny imagenet dataset
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    

import glob
import re
import tensorflow as tf
import random
import numpy as np
import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()

def build_label_dicts():
  label_dict, class_description = {}, {}
  with open('E:/tiny_imagenet/tiny-imagenet-200/wnids.txt', 'r') as f:
    for i, line in enumerate(f.readlines()):
      synset = line[:-1]  # remove \n
      label_dict[synset] = i
  with open('E:/tiny_imagenet/tiny-imagenet-200/words.txt', 'r') as f:
    for i, line in enumerate(f.readlines()):
      synset, desc = line.split('\t') # 同义词
      desc = desc[:-1]  # remove \n
      if synset in label_dict:
        class_description[label_dict[synset]] = desc

  return label_dict, class_description

def load_filenames_labels(mode):
  label_dict, class_description = build_label_dicts()
  filenames_labels = []
  if mode == 'train':
    filenames = glob.glob('E:\\tiny_imagenet\\tiny-imagenet-200\\train\\*\\images\\*.JPEG')
    for filename in filenames:
      match = re.search(r'n\d+', filename)
      label = str(label_dict[match.group()])
      filenames_labels.append((filename, label))
  elif mode == 'val':
    with open('E:\\tiny_imagenet\\tiny-imagenet-200\\val\\val_annotations.txt', 'r') as f:
      for line in f.readlines():
        split_line = line.split('\t')
        filename = 'E:\\tiny_imagenet\\tiny-imagenet-200\\val\\images\\' + split_line[0]
        label = str(label_dict[split_line[1]])
        filenames_labels.append((filename, label))

  return filenames_labels



def parse_fn(filename):
    image_string = tf.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    return image




filenames_labels = load_filenames_labels('train')
filenames = tf.constant([t[0] for t in filenames_labels])
dataset = tf.data.Dataset.from_tensor_slices((filenames))
dataset = dataset.map(parse_fn, num_parallel_calls=8)
dataset = dataset.batch(batch_size=100)


mean_list = []
count = 0
for batch in dataset:
    count = count + 1
    tmp = np.mean(batch,axis =(0,1,2))
    mean_list.append(tmp)
    print(count, tmp)


'''
# Using for loop: too slow(take 15 hours, to 20000 step)
# dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
# means = tf.expand_dims(tf.expand_dims(means, 0), 0)
import time
start_time = time.time()
count = 0
mean_list = []
for name in filenames:
    image_string = tf.read_file(name)
    image = tf.image.decode_jpeg(image_string, channels=3)
    with tf.Session() as sess:
        tmp = sess.run(image)
    mean_tmp = np.mean(tmp,axis =(0,1))
    mean_list.append(mean_tmp)
    count = count + 1
    print('count', count)
    # np.mean(tmp,axis =(0,1))  # average over third channel 
end_time = time.time()
'''
