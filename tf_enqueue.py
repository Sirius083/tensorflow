# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 20:07:17 2018

@author: Sirius
"""

import tensorflow as tf

# This time, let's start with 6 samples of 1 data point
x_input_data = tf.random_normal([6], mean=-1, stddev=4)

# Note that the FIFO queue has still a capacity of 3
q = tf.FIFOQueue(capacity=3, dtypes=tf.float32)

# To check what is happening in this case:
# we will print a message each time "x_input_data" is actually computed
# to be used in the "enqueue_many" operation
x_input_data = tf.Print(x_input_data, data=[x_input_data], message="Raw inputs data generated:", summarize=6)
enqueue_op = q.enqueue_many(x_input_data)

# To leverage multi-threading we create a "QueueRunner"
# that will handle the "enqueue_op" outside of the main thread
# We don't need much parallelism here, so we will use only 1 thread
numberOfThreads = 3 
qr = tf.train.QueueRunner(q, [enqueue_op] * numberOfThreads)
# Don't forget to add your "QueueRunner" to the QUEUE_RUNNERS collection
tf.train.add_queue_runner(qr) 

input_ = q.dequeue() 
input_ = tf.Print(input_, data=[q.size(), input_], message="Nb elements left, input:")

# fake graph: START
y = input_ + 1
# fake graph: END 

# We start the session as usual ...
with tf.Session() as sess:
    # But now we build our coordinator to coordinate our child threads with
    # the main thread
    coord = tf.train.Coordinator()
    # Beware, if you don't start all your queues before runnig anything
    # The main threads will wait for them to start and you will hang again
    # This helper start all queues in tf.GraphKeys.QUEUE_RUNNERS
    threads = tf.train.start_queue_runners(coord=coord)

    # The QueueRunner will automatically call the enqueue operation
    # asynchronously in its own thread ensuring that the queue is always full
    # No more hanging for the main process, no more waiting for the GPU
    sess.run(y)
    print('1')
    sess.run(y)
    print('2')
    sess.run(y)
    print('3')
    sess.run(y)
    print('4')
    sess.run(y)
    print('5')
    sess.run(y)
    print('6')
    sess.run(y)
    print('7')
    sess.run(y)
    print('8')
    sess.run(y)
    print('9')
    sess.run(y)
    print('10')

    # We request our child threads to stop ...
    coord.request_stop()
    # ... and we wait for them to do so before releasing the main thread
    coord.join(threads)
