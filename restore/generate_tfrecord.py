# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 14:00:02 2018

@author: Sirius

参考链接
http://www.cnblogs.com/arkenstone/p/7507261.html

根据图片生成tfrecord数据(ImageNet数据)
sess.close()
tf.reset_default_graph()
"""
'''
#==============================================================================
# 准备数据：imagenet2012的validation set
import os
dir_path = r'E:\ImageNet2012\ImageNet2012\ILSVRC2012_img_val'
images_path = os.listdir(dir_path)
images = [os.path.join(dir_path, path) for path in images_path]
labels_path = r'E:\ImageNet2012\ImageNet2012\ILSVRC2012_validation_ground_truth.txt'
with open(labels_path) as f:
    labels = f.read().splitlines() 
labels = [int(t) for t in labels]

images = images[:100]
labels = labels[:100]

# 生成tfrecord文件
import tensorflow as tf
import cv2

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

train_filename = 'train.tfrecords'

with tf.python_io.TFRecordWriter(train_filename) as tfrecord_writer:  
    for i in range(len(images)):
        # read in image data by tf
        img_data = tf.gfile.FastGFile(images[i], 'rb').read()  # image data type is string
        label = labels[i]
        # get width and height of image
        image_shape = cv2.imread(images[i]).shape
        width = image_shape[1]
        height = image_shape[0]
        # create features
        feature = {'train/image': _bytes_feature(img_data),
                   'train/label': _int64_feature(label),  # label: integer from 0-N
                   'train/height': _int64_feature(height), 
                   'train/width': _int64_feature(width)}
        # create example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        # serialize protocol buffer to string
        tfrecord_writer.write(example.SerializeToString())
    # tfrecord_writer.close()
'''
#==============================================================================
# 读取tfrecord文件 
import tensorflow as tf
import matplotlib.pyplot as plt

data_path = 'train.tfrecords'

with tf.Session() as sess:
    # feature key and its data type for data restored in tfrecords file
    feature = {'train/image': tf.FixedLenFeature([], tf.string),
                     'train/label': tf.FixedLenFeature([], tf.int64),
                     'train/height': tf.FixedLenFeature([], tf.int64),
                     'train/width': tf.FixedLenFeature([], tf.int64)}
    # define a queue base on input filenames
    filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)
    # define a tfrecords file reader
    reader = tf.TFRecordReader()
    # read in serialized example data
    _, serialized_example = reader.read(filename_queue)
    # decode example by feature
    features = tf.parse_single_example(serialized_example, features=feature)
    image = tf.image.decode_jpeg(features['train/image'])
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)  # convert dtype from unit8 to float32 for later resize
    label = tf.cast(features['train/label'], tf.int64)
    height = tf.cast(features['train/height'], tf.int32)
    width = tf.cast(features['train/width'], tf.int32)
    # restore image to [height, width, 3]
    image = tf.reshape(image, [height, width, 3])
    # resize
    image = tf.image.resize_images(image, [224, 224])
    # create batch
    # # capacity是队列的最大容量，num_threads是dequeue后最小的队列大小，num_threads是进行队列操作的线程数。
    images, labels = tf.train.shuffle_batch([image, label], batch_size=10, capacity=30, num_threads=1, min_after_dequeue=10) 

    # initialize global & local variables
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    # create a coordinate and run queue runner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for batch_index in range(3):
        batch_images, batch_labels = sess.run([images, labels])
        for i in range(10):
            plt.imshow(batch_images[i, ...])
            plt.show()
            print ("Current image label is: ", batch_labels[i])
    # close threads
    coord.request_stop()
    coord.join(threads)
    # sess.close()

