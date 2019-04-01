#!/usr/bin/env python
# -*- coding: utf-8 -*-
# tensorflow 1.9
# python 3.6

import random
import os
import numpy as np
import tensorflow as tf
from attr_converter import *


# load graph & print node name and shape
graph_pb_path = r'E:\denseNet\densenet_imagenet\pretrained\tf-densenet121\output.pb'
tf.reset_default_graph()
config = tf.ConfigProto(allow_soft_placement = True)

with tf.gfile.GFile(graph_pb_path, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
with tf.Graph().as_default() as graph:
     tf.import_graph_def(graph_def, name="")
     print('load graph pb done')

     # print all graph node name and shape
     all_node = [n for n in tf.get_default_graph().as_graph_def(add_shapes=True).node]
     for node in all_node:
         node_shape = node.attr['_output_shapes'].list.shape[0].dim
         shape = []
         for i in range(len(node_shape)):
             shape = shape + [node_shape[i].size]
         print('{:70s}'.format(node.name), shape)
         

