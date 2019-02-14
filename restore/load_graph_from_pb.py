import tensorflow as tf
from tensorflow.python.platform import gfile
GRAPH_PB_PATH = r'E:\transfer_tiny_imagenet\inception\classify_image_graph_def.pb'


with tf.Session() as sess:    
    print("load graph")
    with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
         graph_def = tf.GraphDef()
         graph_def.ParseFromString(f.read())
         sess.graph.as_default()
         tv = tf.import_graph_def(graph_def,return_elements=['pool_3:0']) # 从graph中导入tensor的定义
         print('tv', tv)
