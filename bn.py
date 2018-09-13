# Batch normalization parameter number
# BN 讲解比较好的一篇文章
# https://blog.csdn.net/hjimce/article/details/50866313
# fully connected bn
# data = tf.zeros((10,200))
# data2 = tf.layers.batch_normalization(inputs=data, axis = 1, momentum=0.997, epsilon=1e-5, center=True,scale=True)

data = tf.zeros((10, 64, 64, 3)) # convolutional bn
data1 = tf.layers.conv2d(data, filters=16, kernel_size=3,padding='same')
data2 = tf.layers.batch_normalization(inputs=data1, axis = 1, momentum=0.997, epsilon=1e-5, center=True,scale=True)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    allvar = tf.global_variables() # all global variables
    alltrainvar = tf.trainable_variables() # all trainable variables
    # print all variables
    print('all variables =========== ')
    for var in allvar:
        print(var)
    
    print('all trainable variables ===========')
    for var in alltrainvar:
        print(var)


'''
all variables ===========
<tf.Variable 'batch_normalization/gamma:0' shape=(64,) dtype=float32_ref>
<tf.Variable 'batch_normalization/beta:0' shape=(64,) dtype=float32_ref>
<tf.Variable 'batch_normalization/moving_mean:0' shape=(64,) dtype=float32_ref>
<tf.Variable 'batch_normalization/moving_variance:0' shape=(64,) dtype=float32_ref>
all trainable variables ===========
<tf.Variable 'batch_normalization/gamma:0' shape=(64,) dtype=float32_ref>
<tf.Variable 'batch_normalization/beta:0' shape=(64,) dtype=float32_ref>
'''

'''
all variables ===========
<tf.Variable 'conv2d/kernel:0' shape=(3, 3, 3, 16) dtype=float32_ref>
<tf.Variable 'conv2d/bias:0' shape=(16,) dtype=float32_ref>
<tf.Variable 'batch_normalization/gamma:0' shape=(64,) dtype=float32_ref>
<tf.Variable 'batch_normalization/beta:0' shape=(64,) dtype=float32_ref>
<tf.Variable 'batch_normalization/moving_mean:0' shape=(64,) dtype=float32_ref>
<tf.Variable 'batch_normalization/moving_variance:0' shape=(64,) dtype=float32_ref>
all trainable variables ===========
<tf.Variable 'conv2d/kernel:0' shape=(3, 3, 3, 16) dtype=float32_ref>
<tf.Variable 'conv2d/bias:0' shape=(16,) dtype=float32_ref>
<tf.Variable 'batch_normalization/gamma:0' shape=(64,) dtype=float32_ref>
<tf.Variable 'batch_normalization/beta:0' shape=(64,) dtype=float32_ref>
'''
