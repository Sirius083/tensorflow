# https://github.com/tensorflow/tensorflow/issues/8246
# 对输入hight和width进行重复
import tensorflow as tf
import numpy as np

arr = np.array([8,9,10,11,1,2,3,4]).reshape(2,2,2)
# arr = np.array([8,9,10,11]).reshape(2,2)
'''
arr1 = np.repeat(arr, [2], axis=1)
arr2 = np.repeat(arr1, [3], axis=2)
#  查看array每个axis的数字
arr[0,:,:]
arr[:,0,:]
arr[:,:,0]
'''
repeats = [1,2,3]
# repeats = [2,3]
tensor = tf.constant(arr, dtype=tf.float32) # (2,2)
output_shape = arr.shape * np.array(repeats)

expanded_tensor = tf.expand_dims(tensor, -1) # (2,2,1)
multiples = [1] + repeats # [1,2,3]
tiled_tensor = tf.tile(expanded_tensor, multiples = multiples) # [2,4,3]
# repeated_tesnor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats) # result in unknown shape
repeated_tesnor = tf.reshape(tiled_tensor, output_shape)

sess = tf.Session()
t = sess.run(tensor)
te = sess.run(expanded_tensor)
tt  = sess.run(tiled_tensor)
tmp = sess.run(repeated_tesnor)
print('tensor\n', t)
print('expand tensor\n', te)
print('tiled tensor\n',tt)
print('repeated tensor\n', tmp)
sess.close()
  
