# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 09:01:00 2018

@author: Sirius

在一个session同时使用train和validation的pipeline
https://github.com/tensorflow/tensorflow/issues/15448
You should use the feedable iterator if you want to switch between two iterators without resetting.
"""
import tensorflow as tf
# Define training and validation datasets with the same structure.
training_dataset = tf.data.Dataset.range(100).shuffle(200).repeat(2)
validation_dataset = tf.data.Dataset.range(50).shuffle(100).repeat(2)
# training_dataset = tf.data.Dataset.range(100).repeat(2)
# validation_dataset = tf.data.Dataset.range(50).repeat(2)
# A feedable iterator is defined by a handle placeholder and its structure. We
# could use the `output_types` and `output_shapes` properties of either
# `training_dataset` or `validation_dataset` here, because they have
# identical structure.
handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(
    handle, training_dataset.output_types, training_dataset.output_shapes)
next_element = iterator.get_next()

# You can use feedable iterators with a variety of different kinds of iterator
training_iterator = training_dataset.make_one_shot_iterator()
validation_iterator = validation_dataset.make_initializable_iterator()

with tf.Session() as sess:
    # The `Iterator.string_handle()` method returns a tensor that can be evaluated
    # and used to feed the `handle` placeholder.
    training_handle = sess.run(training_iterator.string_handle())
    validation_handle = sess.run(validation_iterator.string_handle())
    # Loop forever, alternating between training and validation.
    for i in range(20):
        print("######################## ", i)
        i += 1
        # Run 10 steps using the training dataset. Note that the training dataset is
        # 2 * the original set, i.e. we run 2 epochs (see .repeat() argument), and we resume from where
        # we left off in the previous `while` loop iteration.
        for _ in range(10):
            nel = sess.run(next_element, feed_dict={handle: training_handle})
            print("train: ", type(nel), nel)

        # Run one pass over the validation dataset.
        sess.run(validation_iterator.initializer)
        for _ in range(5):
            nel = sess.run(next_element, feed_dict={handle: validation_handle})
            print("valid: ", type(nel), nel)
'''       
# tensorflow 官网给出的教程 
train_iterator = tf.data.Dataset(...).make_one_shot_iterator()
train_iterator_handle = sess.run(train_iterator.string_handle())

test_iterator = tf.data.Dataset(...).make_one_shot_iterator()
test_iterator_handle = sess.run(test_iterator.string_handle())

handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(
    handle, train_iterator.output_types)

next_element = iterator.get_next()

loss = f(next_element)

train_loss = sess.run(loss, feed_dict={handle: train_iterator_handle})
test_loss = sess.run(loss, feed_dict={handle: test_iterator_handle})
'''
