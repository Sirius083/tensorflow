# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 16:01:22 2018

@author: Sirius

https://blog.csdn.net/u012564409/article/details/78247216?utm_source=debugrun&utm_medium=referral
name_scope和variable_scope之间的区别/异同
"""

'''
1 在tf.name_scope下时，tf.get_variable()创建的变量名不受name_scope的影响，
  而在未指定共享变量时，如果重名就会报错
2.在tf.name_scope下时，tf.Variable()会自动检测有没有变量重名，如果有则会自行处理。
  即给变量加上var_1和var_2
'''
import tensorflow as tf
with tf.name_scope('name_scope_x'):
    var1 = tf.get_variable(name='var1', shape=[1], dtype=tf.float32)
    var3 = tf.Variable(name='var2', initial_value=[2], dtype=tf.float32)
    var4 = tf.Variable(name='var2', initial_value=[2], dtype=tf.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(var1.name, sess.run(var1))
    print(var3.name, sess.run(var3))
    print(var4.name, sess.run(var4))

# 输出结果：
# var1:0 [-0.30036557]   可以看到前面不含有指定的'name_scope_x'
# name_scope_x/var2:0 [ 2.]
# name_scope_x/var2_1:0 [ 2.]  可以看到变量名自行变成了'var2_1'，避免了和'var2'冲突

'''
2 使用tf.get_variable()创建变量，且没有设置共享变量，重名时会报错。
'''
import tensorflow as tf
with tf.name_scope('name_scope_1'):
    var1 = tf.get_variable(name='var1', shape=[1], dtype=tf.float32)
    var2 = tf.get_variable(name='var1', shape=[1], dtype=tf.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(var1.name, sess.run(var1))
    print(var2.name, sess.run(var2))

# ValueError: Variable var1 already exists, disallowed. Did you mean 
# to set reuse=True in VarScope? Originally defined at:
# var1 = tf.get_variable(name='var1', shape=[1], dtype=tf.float32)

'''
3 共享变量方法，（要共享变量就要使用tf.get_variable(<variable_name>）
'''
import tensorflow as tf
with tf.variable_scope('variable_scope_y') as scope:
    var1 = tf.get_variable(name='var1', shape=[1], dtype=tf.float32)
    scope.reuse_variables()  # 设置共享变量
    var1_reuse = tf.get_variable(name='var1')
    var2 = tf.Variable(initial_value=[2.], name='var2', dtype=tf.float32)
    var2_reuse = tf.Variable(initial_value=[2.], name='var2', dtype=tf.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(var1.name, sess.run(var1))
    print(var1_reuse.name, sess.run(var1_reuse))
    print(var2.name, sess.run(var2))
    print(var2_reuse.name, sess.run(var2_reuse))

# 输出结果：
# variable_scope_y/var1:0 [-1.59682846]
# variable_scope_y/var1:0 [-1.59682846]   可以看到变量var1_reuse重复使用了var1
# variable_scope_y/var2:0 [ 2.]
# variable_scope_y/var2_1:0 [ 2.]

# 共享变量定义方式 2
with tf.variable_scope('foo') as foo_scope:
    v = tf.get_variable('v', [1])
with tf.variable_scope('foo', reuse=True): # variable_scope: foo
    v1 = tf.get_variable('v')
assert v1 == v


# 共享变量定义方式 3
with tf.variable_scope('foo') as foo_scope:
    v = tf.get_variable('v', [1])
with tf.variable_scope(foo_scope, reuse=True): # variable_scope: foo_scope
    v1 = tf.get_variable('v')
assert v1 == v

#=======================================================
# name_scope vs varibale_scope
# 1. tf.name_scope() 用于管理一个图例各种ops, 每个namespace下可以定义各种op或者子namespace
#    实现一种层次化有条理的管理，避免各个op之间的冲突
# 2. tf.variable_scope() 与 tf.name_scope() 配合使用，用于管理一个graph中各种变量的名字
#    避免变量之间的命名冲突，tf.variable_scope() 允许在一个variable_scope下共享变量
# 3. 区别：name_scope只能管住ops的名字，不能管住variables的名字
with tf.variable_scope("foo"):
   with tf.name_scope("bar"):
       v = tf.get_variable("v", [1])
       x = 1.0 + v
print(v.name)      # foo/v:0
print(x.op.name)   # foo/bar/add

# assert v.name == "foo/v:0"
# assert x.op.name == "foo/bar/add"


