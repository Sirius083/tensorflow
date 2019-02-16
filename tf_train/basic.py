# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 11:12:19 2019

@author: Sirius
"""

# disable GPU useage
import os
os.environ["CUDA_VISABLE_DEVICES"] = "-1"
import tensorflow as tf

# count total parameters
def count_params():
    total_params=0
    for variable in tf.trainable_variables():
        shape=variable.get_shape()
        params=1
        for dim in shape:
            params=params*dim.value
        total_params+=params
    print("Total training params: %.2fM" % (total_params / 1e6))

# create global step
# do not need to use current epoch and bathc number
train_op = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(loss_tensor,global_step=tf.train.create_global_step())
with tf.Session() as sess:
    sess.run([loss, accuracy, tf.train.get_global_step()])



# tensorflow continue training from stored model
# Note: tf.train.Saver must be created 
#       1. after the variables that you want to restore(or save)
#       2. in the same graph as those variables

# saver restore example
with tf.Graph().as_default() as g:
	images, labels = Process.eval_inputs(eval_data = eval_data)
	forward_propgation_results = Process.forward_propagation(images) # define variables
  	init_op = tf.initialize_all_variables()
	saver = tf.train.Saver()
	top_k_op = tf.nn.in_top_k(forward_propgation_results, labels, 1)

with tf.Session(graph=g) as sess:
	sess.run(init_op)
	saver.restore(sess, eval_dir)
	writer = tf.summary.FileWriter('board_beginner')
	writer.add_graph(g)

	for step in range(step_num):
		batch_x, batch_y = ...
		sess.run(train_step)

		if step % 20 == 0:
			acc = sess.run()
			writer.add_summary(acc, step)

	saver.save(sess,'model_beginner')



# FileWriter save & restore
# logging train and test accuracy in the same graph
# 1. variable defination and save
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

# 2. merge all summaries
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
tf.global_variables_initializer().run()

for i in range(steps):
    if i % 10 == 0:
	   summary,acc = sess.run([merged, accuracy], feed_dict = [])
	   test_writer.add_summary(summary,i)
	   print('accuracy at step %s: %s ', (i,accuracy))
    else:
    	summary, _ = sess.run([merged, train_step], feed_dict = [])
    	train_writer.add_summary(summary, i)

train_writer.close()
test_writer.close()

# Note: 
# average summaries over batches (in validation)
# do averaging of your measure in python and create a new summary object for each mean
# test on 10000 samples --> memory out of use 

accuracies = []
for batch in validation_set:
	accuracies.append([sess.run(acc)])

accuracy = np.mean(accuracies)

# create a new summary object
summary = tf.Summary()
summary.value.add(tag = "%sAccuracy" % prefix, simple_value = accuracy)

# add it to the tensorboard summary writer
summary_writer.add_summary(summary, global_step)


# change learning rate
for epoch in range(1, total_epoches+1):
    if epoch == total_epoches/2 : lr=lr*0.1
    if epoch == total_epoches*3/4 : lr=lr*0.1


