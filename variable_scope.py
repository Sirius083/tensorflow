# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 11:38:35 2018

@author: Sirius

using variables defined in another function

如果明确定义了指定的scope
在variable_scope中使用
则不会使用当前variable_scope的名字
"""

import tensorflow as tf

top_scope = tf.get_variable_scope()

def create_variables():
    with tf.variable_scope('level2') as scope:
        var = tf.get_variable('var2', shape = [2])
    return var

with tf.variable_scope('level1') as scope:
     var1 = tf.get_variable('var1', shape = [1])
     
     with tf.variable_scope(top_scope) as scope:
          var2 = create_variables()
          var3 = tf.get_variable('var3', shape = [3])


# 定义了scope, 后面引用指定的scope
with tf.variable_scope('scope1') as scope_1:
     var4 = tf.get_variable('var4', shape = [4])

# var5 与 var4 共用同一个scope
with tf.variable_scope(scope_1) as scope:
     var5 = tf.get_variable('var5', shape = [5])      
          
init = tf.global_variables_initializer()

with tf.Session() as sess:
     sess.run(init)
     allvar = tf.global_variables()
     for var in allvar:
         print(var)
