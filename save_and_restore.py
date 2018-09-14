# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 19:15:10 2018

@author: Sirius

参考资料1：

参考资料2：
https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
"""

'''
meta graph: save model graph  --> .meta
checkpoint: parameters        --> .data-00000-of-00001 & .index
'''
import tensorflow as tf
import os

# =============================================================================
# save model variables
# Create some variables.
v1 = tf.get_variable("v1", shape=[3], initializer = tf.zeros_initializer)
v2 = tf.get_variable("v2", shape=[5], initializer = tf.zeros_initializer)

inc_v1 = v1.assign(v1+1)
dec_v2 = v2.assign(v2-1)

# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, initialize the variables, do some work, and save the
# variables to disk.
with tf.Session() as sess:
  sess.run(init_op)
  # Do some work with the model.
  inc_v1.op.run()
  dec_v2.op.run()
  # Save the variables to disk.
  path_1 = os.path.join(os.getcwd(), "tmp\\model.ckpt")
  save_path = saver.save(sess, path_1)
  print("Model saved in path: %s" % save_path)

# =============================================================================
# restore variables
tf.reset_default_graph()

# Create some variables.
v1 = tf.get_variable("v1", shape=[3])
v2 = tf.get_variable("v2", shape=[5])

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess, path_1)
  print("Model restored.")
  # Check the values of the variables
  print("v1 : %s" % v1.eval())
  print("v2 : %s" % v2.eval())

# =============================================================================
# choose variables to save and restore
tf.reset_default_graph()
# Create some variables.
v1 = tf.get_variable("v1", [3], initializer = tf.zeros_initializer)
v2 = tf.get_variable("v2", [5], initializer = tf.zeros_initializer)

# Add ops to save and restore only `v2` using the name "v2"
saver = tf.train.Saver({"v2": v2})

# Use the saver object normally after that.
with tf.Session() as sess:
  # Initialize v1 since the saver will not.
  v1.initializer.run()
  saver.restore(sess, "/tmp/model.ckpt")

  print("v1 : %s" % v1.eval())
  print("v2 : %s" % v2.eval())

# =============================================================================
# inspect variables in a checkpoint
# import the inspect_checkpoint library
from tensorflow.python.tools import inspect_checkpoint as chkp

# print all tensors in checkpoint file
chkp.print_tensors_in_checkpoint_file(path_1, tensor_name='', all_tensors=True)

# tensor_name:  v1
# [ 1.  1.  1.]
# tensor_name:  v2
# [-1. -1. -1. -1. -1.]

# print only tensor v1 in checkpoint file
chkp.print_tensors_in_checkpoint_file(path_1, tensor_name='v1', all_tensors=False)

# tensor_name:  v1
# [ 1.  1.  1.]

# print only tensor v2 in checkpoint file
chkp.print_tensors_in_checkpoint_file(path_1, tensor_name='v2', all_tensors=False)

# tensor_name:  v2
# [-1. -1. -1. -1. -1.]

# =============================================================================
# save a whole model
saver = tf.train.Saver()
saver.save(sess, 'my-test-model')

# complete version
import tensorflow as tf
w1 = tf.Variable(tf.random_normal(shape=[2]),name='w1')
w2 = tf.Variable(tf.random_normal(shape=[5]),name='w2')
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver.save(sess,'my_test_model')

# save model after 1000 iterations
saver.save(sess,'my_test_model',step=1000) 

# save for future iterations
saver.save(sess, 'my-model',global_step = step, write_meta_graph = False)

# keep only 4 latest model/every two hours
saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours = 2)

# only save part variables
saver = tf.train.Saver([w1,w2])

# =============================================================================
# import a pre-trained model
# 1. load graph
saver = tf.train.import_meta_graph('my_test_model-1000.meta')

# 2. load trained values
with tf.Session() as sess:
	new_saver = tf.train.import_meta_graph('my_test_model-1000.meta')
	new_saver.restore(sess,tf.train.latest_checkpoint('./'))
	print(sess.run('w1:0'))

# =============================================================================
# work with restored models

import tensorflow as tf

#Prepare to feed input, i.e. feed_dict and placeholders
w1 = tf.placeholder("float", name="w1")
w2 = tf.placeholder("float", name="w2")
b1= tf.Variable(2.0,name="bias")
feed_dict ={w1:4,w2:8}

#Define a test operation that we will restore
w3 = tf.add(w1,w2)
w4 = tf.multiply(w3,b1,name="op_to_restore")
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#Create a saver object which will save all the variables
saver = tf.train.Saver()

#Run the operation by feeding input
print sess.run(w4,feed_dict)
#Prints 24 which is sum of (w1+w2)*b1 

#Now, save the graph
saver.save(sess, 'my_test_model',global_step=1000)


# =============================================================================
# working with resotred models
import tensorflow as tf
# define a feed input, i.e: feed_dict and placeholders
w1 = tf.placeholder('float', name = 'w1')
w2 = tf.placeholder('float', name = 'w2')
b1 = tf.Variable(2.0, name = 'bias')
feed_dict = {w1:4, w2:8}

# define a test operation that we will restore
w3 = tf.add(w1,w2)
w4 = tf.multiply(w3,b1,name = 'op_to_restore')
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# create a saver object which will save all the variables
saver = tf.train.Saver()

# run the operation by feeding input
print(sess.run(w4, feed_dict))

# save the graph
saver.save(sess, 'my_test_model', global_step = 1000)


# get reference to these saved operations and placeholder variables via graph.get_tensor_by_name()
# access saved variable/tensor/placeholders
w1 = graph.get_tensor_by_name("w1:0")
op_to_restore = graph.get_tensor_by_name("op_to_restore:0")

# run the same network with different data, pass new data via feed_dict
sess = tf.Session()
saver = tf.train.import_meta_graph('my_test_model.meta')
saver.restore(sess, tf.train.latest_checkpoint('./'))


graph = tf.get_default_graph()
w1 = graph.get_tensor_by_name("w1:0")
w2 = graph.get_tensor_by_name("w2:0")

# create feed_dict to feed new data
feed_dict = {w1:13.0, w2:17.0}

# access the operation you want to run
op_to_restore = graph.get_tensor_by_name("op_to_restore:0")
print(sess.run(op_to_restore, feed_dict))

# =============================================================================
# add more operaion and retrain the graph

import tensorflow as tf

sess=tf.Session()    
#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('my_test_model-1000.meta')
saver.restore(sess,tf.train.latest_checkpoint('./'))


# Now, let's access and create placeholders variables and
# create feed-dict to feed new data

graph = tf.get_default_graph()
w1 = graph.get_tensor_by_name("w1:0")
w2 = graph.get_tensor_by_name("w2:0")
feed_dict ={w1:13.0,w2:17.0}

#Now, access the op that you want to run. 
op_to_restore = graph.get_tensor_by_name("op_to_restore:0")

#Add more to the current graph
add_on_op = tf.multiply(op_to_restore,2)

print sess.run(add_on_op,feed_dict)
#This will print 120.



# =============================================================================
# retrain the last fc layer for vgg pre-trained model
saver = tf.train.import_meta_graph('vgg.meta')
# Access the graph
graph = tf.get_default_graph()
## Prepare the feed_dict for feeding data for fine-tuning 

#Access the appropriate output for fine-tuning
fc7= graph.get_tensor_by_name('fc7:0')

#use this if you only want to change gradients of the last layer
fc7 = tf.stop_gradient(fc7) # It's an identity function
fc7_shape= fc7.get_shape().as_list()

new_outputs=2
weights = tf.Variable(tf.truncated_normal([fc7_shape[3], num_outputs], stddev=0.05))
biases = tf.Variable(tf.constant(0.05, shape=[num_outputs]))
output = tf.matmul(fc7, weights) + biases
pred = tf.nn.softmax(output)

# Now, you run this with fine-tuning data in sess.run()













