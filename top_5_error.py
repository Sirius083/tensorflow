# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 22:22:57 2018

@author: Sirius

判断分类结果是否在top_3
"""

# 作为验证的例子
import numpy as np
import tensorflow as tf

a = np.random.rand(4,5)
b = a/a.sum(axis=1,keepdims=1) # 对行进行归一化
logits = tf.constant(b)
labels = tf.constant([3,2,1,4])

logits = tf.cast(logits, tf.float32)
top_3_bool = tf.nn.in_top_k(predictions=logits, targets=labels, k=3)
acc_3 = tf.reduce_mean(tf.cast(top_3_bool, tf.float32))

with tf.Session() as sess:
    top_3_bool_eval = sess.run(top_3_bool)
    acc_3_eval = sess.run(acc_3)

print(b)
print('top_3_bool_eval',top_3_bool_eval)
print('acc_3_eval',acc_3_eval)
