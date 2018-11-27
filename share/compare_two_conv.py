import tensorflow as tf
g = tf.get_default_graph()
x = tf.random_normal(shape=[2, 32, 32, 3], seed = 7)
y = tf.constant([1,2])
label = tf.one_hot(y, depth = 10)

# 这两层用到了相同的卷积层，共享变量
conv1 = tf.layers.conv2d(x, 3, [5, 5], padding='SAME', reuse=None, name='conv', 
                         kernel_initializer = tf.contrib.layers.xavier_initializer(seed = 1))
# conv2 = tf.layers.conv2d(conv1, 3, [5, 5], padding='SAME', reuse=True, name='conv')
output = tf.reshape(conv1, [2, -1])
logits = tf.layers.dense(output, 10,
                         kernel_initializer = tf.contrib.layers.xavier_initializer(seed = 1))

# tf.nn.softmax_cross_entropy_with_logits: 相当于将两部合起来了
# y_out: softmax(logits)
# sum(y_true*log(y_out))
entropy = tf.nn.softmax_cross_entropy_with_logits(labels = label, logits = logits)
loss = tf.reduce_mean(entropy, name = "loss")


# 获得图中tensor的名字
conv_kernel = g.get_tensor_by_name('conv/kernel:0')  # (5,5,3,3)
conv_bias = g.get_tensor_by_name('conv/bias:0')       #  (3,)
dense_kernel = g.get_tensor_by_name('dense/kernel:0') # (3072,10)
dense_bias = g.get_tensor_by_name('dense/bias:0')     # (10,)

# =============================================================================
# 梯度计算
# 从输入开始进行卷积，fc计算

output1 = tf.nn.conv2d(x, conv_kernel, [1,1,1,1], padding = "SAME") + conv_bias
output2 = tf.nn.conv2d(output1, conv_kernel, [1,1,1,1], padding = "SAME") + conv_bias
output2 = tf.reshape(output2,[2,-1])
output3 = tf.matmul(output2, dense_kernel) + dense_bias
output4 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = label, logits = output3))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
LOSS = sess.run(loss)
# OUTPUT = sess.run(output4)
print(tf.global_variables())

