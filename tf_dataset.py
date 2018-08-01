# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 17:19:32 2018

@author: Sirius
"""

# https://www.tensorflow.org/performance/datasets_performance
# https://www.tensorflow.org/guide/datasets
# tensorflow QueueRunners: https://learningai.io/projects/2017/03/25/tf-queue-runner.html

# How to optimise your input pipeline with queues and multi-threading
# https://blog.metaflow.fr/tensorflow-how-to-optimise-your-input-pipeline-with-queues-and-multi-threading-e7c3874157e0

import tensorflow as tf
dataset = dataset.map(map_func = parse_fn) 
# num_parallel_calls: task manager 中CPU的 Logical processors
# batch: 不同的picture同时预处理 
dataset = dataset.map(map_func = parse_fn, num_parallel_calls = FLAGS.num_parallel_calls)
dataset = dataset.batch(batch_size = FLAGS.batch_size)

dataset = dataset.apply(tf.contrib.data.map_and_batch(map_func = parse_fn, batch_size = FLAGS.batch_size))



dataset = dataset.batch(batch_size = FLAGS.batch_size)
dataset = dataset.prefetch(buffer_size = FLAGS.prefetch_buffer_size)
return dataset

# number of 
