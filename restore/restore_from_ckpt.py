# Note clear_devices = True
import os

model_dir = r'E:\resnet\models-master\official\resnet\train_dir\imagenet_share_34'
meta_path = os.path.join(model_dir, 'model.ckpt-1242270.meta')
model_path = os.path.join(model_dir, 'model.ckpt-1242270')

os.chdir(model_dir)
import tensorflow as tf
# import tensorflow.contrib
# import tensorflow.contrib.resampler

with tf.Session() as sess:
    saver = tf.train.import_meta_graph(meta_path,clear_devices = True)
    saver.restore(sess, model_path)
