#!/usr/bin/env python
# -*- coding: utf-8 -*-
# evaluation one densenet-121: 8-min (batch_size=250)

import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
import numpy as np
import tensorflow as tf
from densenet_preprocessing import *
from utility import *


_RESIZE_SIDE_MIN = 256
_RESIZE_SIDE_MAX = 512
_VAL_NUM = 50000
_BATCH_SIZE = 250

def get_image_preprocessed(image_path):
    with open(image_path, 'rb') as f:
        contents = f.read()

    image = tf.image.decode_jpeg(contents, channels = 3)

    image_preprocessed = preprocess_image(image, 224, 224, is_training=False,
                                        resize_side_min=_RESIZE_SIDE_MIN,
                                        resize_side_max=_RESIZE_SIDE_MAX)
    image_preprocessed = tf.expand_dims(image_preprocessed, axis = 0)

    with tf.Session() as sess:
        image_preprocessed = sess.run(image_preprocessed)

    print('*******************************')
    print('preprocess image done', image_path)
    return image_preprocessed

def get_queue_batch(image_dir, batch_size):
    all_images = os.listdir(image_dir)
    filelist = [os.path.join(image_dir, path) for path in all_images]
    # Note: set shuffle=False to preserve order
    file_queue = tf.train.string_input_producer(filelist, shuffle = False)
    reader = tf.WholeFileReader()
    key,value = reader.read(file_queue)
    image = tf.image.decode_jpeg(value, channels = 3)
    image = preprocess_image(image, output_height = 224, output_width = 224, 
                     is_training=False,resize_side_min=_RESIZE_SIDE_MIN,
                     resize_side_max=_RESIZE_SIDE_MAX)
    image_batch = tf.train.batch([image],batch_size=_BATCH_SIZE)
    return image_batch


def get_graph(graph_pb_path):
    tf.reset_default_graph()
    config = tf.ConfigProto(allow_soft_placement = True)

    with tf.gfile.GFile(graph_pb_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
    # print('*******************************')
    print('load graph pb done')
    return graph


graph_pb_path = r'E:\denseNet\densenet_imagenet\pretrained\tf-densenet121\output.pb'
image_dir = r'E:\ImageNet2012\ILSVRC2012_img_val'
 
import numpy as np
tf.reset_default_graph()
config = tf.ConfigProto(allow_soft_placement = True)

with tf.gfile.GFile(graph_pb_path, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name="")
    image_batch = get_queue_batch(image_dir, 100)

# layer name
input_name  = "Placeholder:0"
output_name = "densenet121/predictions/Softmax:0"
inputs  = graph.get_tensor_by_name(input_name)
outputs = graph.get_tensor_by_name(output_name)


import time
s_time = time.time()
results = []

with tf.Session(graph = graph) as sess:
   coord = tf.train.Coordinator()
   threads = tf.train.start_queue_runners(sess,coord=coord)

   total_epoch = int(_VAL_NUM/_BATCH_SIZE)
   for i in range(total_epoch):
     # add image_value to evaluation pipiline  
     image_value = sess.run(image_batch)
     res =  sess.run(outputs, {inputs: image_value})
     print(i, res.shape)
     results = results + [res]

   coord.request_stop()
   coord.join(threads)

print('***** evaluate neuron on all image total time', (time.time() - s_time)/60)

results = np.array(results) # (300,100,1000)
results = results.reshape(results.shape[0] * results.shape[1], results.shape[2])
np.save('densenet_121_results.npy', results)       # save
print('validation results saved')

# res_load = np.load('densenet_121_results.npy') # load
