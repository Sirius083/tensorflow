"""
License: Apache-2.0
Author: Suofei Zhang | Hang Yu
E-mail: zhangsuofei at njupt.edu.cn | hangyu5 at illinois.edu
code from https://github.com/www0wwwjs1/Matrix-Capsules-EM-Tensorflow
Add some explanation
too hard to understand


Question:
文章的bias没有体现
特征图的某一部分是检测所有类型的entity还是某一种类型的entity

whole network as regular convolutional nework, see each capsule in the network as a matrix/vector


PrimaryCapsule --> ConvCaps & ConvCaps --> ConvCaps 中间不定义bias是由于
Relu Conv1 --> Primary Caps 是有1X1 卷积显式定义的，后面的activation是通过动态路由计算出来的

EM methods fits datapoints into a mixture of Gaussian models
E-step: determines the assignment probability r_ij of each datapoint to a parent capsule
        (based on Gaussian model (mu,sigma) and the new a_j)
M-step: recalculate the Gaussian model's value (mu,sigma) based on r_ij

last a_j will be the parent capsule's output


V_ij = M_i * W_ij
poses, activations are calculated by em-routing algorithm

all capsules of the same type are extracting the same entity at different positions
share the transformation matrices between different positions of the same capsule type
and add the scaled coordinate(row,column)
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
from config import cfg
import numpy as np

def cross_ent_loss(output, x, y):
    loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=output)
    loss = tf.reduce_mean(loss)
    num_class = int(output.get_shape()[-1])
    data_size = int(x.get_shape()[1])

    # reconstruction loss
    y = tf.one_hot(y, num_class, dtype=tf.float32)
    y = tf.expand_dims(y, axis=2)
    output = tf.expand_dims(output, axis=2)
    output = tf.reshape(tf.multiply(output, y), shape=[cfg.batch_size, -1])
    tf.logging.info("decoder input value dimension:{}".format(output.get_shape()))

    with tf.variable_scope('decoder'):
        output = slim.fully_connected(output, 512, trainable=True)
        output = slim.fully_connected(output, 1024, trainable=True)
        output = slim.fully_connected(output, data_size * data_size,
                                      trainable=True, activation_fn=tf.sigmoid)

        x = tf.reshape(x, shape=[cfg.batch_size, -1])
        reconstruction_loss = tf.reduce_mean(tf.square(output - x))

    # regularization loss
    regularization = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    # loss+0.0005*reconstruction_loss+regularization#
    loss_all = tf.add_n([loss] + [0.0005 * reconstruction_loss] + regularization)

    return loss_all, reconstruction_loss, output


def spread_loss(output, pose_out, x, y, m):
    """
    # check NaN
    # See: https://stackoverflow.com/questions/40701712/how-to-check-nan-in-gradients-in-tensorflow-when-updating
    output_check = [tf.check_numerics(output, message='NaN Found!')]
    with tf.control_dependencies(output_check):
    """

    num_class = int(output.get_shape()[-1])
    data_size = int(x.get_shape()[1])

    y = tf.one_hot(y, num_class, dtype=tf.float32)

    # spread loss
    output1 = tf.reshape(output, shape=[cfg.batch_size, 1, num_class])
    y = tf.expand_dims(y, axis=2)
    at = tf.matmul(output1, y)
    """Paper eq(5)."""
    loss = tf.square(tf.maximum(0., m - (at - output1)))
    loss = tf.matmul(loss, 1. - y)
    loss = tf.reduce_mean(loss)

    # reconstruction loss
    # pose_out = tf.reshape(tf.matmul(pose_out, y, transpose_a=True), shape=[cfg.batch_size, -1])
    pose_out = tf.reshape(tf.multiply(pose_out, y), shape=[cfg.batch_size, -1])
    tf.logging.info("decoder input value dimension:{}".format(pose_out.get_shape()))

    with tf.variable_scope('decoder'):
        pose_out = slim.fully_connected(pose_out, 512, trainable=True, weights_regularizer=tf.contrib.layers.l2_regularizer(5e-04))
        pose_out = slim.fully_connected(pose_out, 1024, trainable=True, weights_regularizer=tf.contrib.layers.l2_regularizer(5e-04))
        pose_out = slim.fully_connected(pose_out, data_size * data_size,
                                        trainable=True, activation_fn=tf.sigmoid, weights_regularizer=tf.contrib.layers.l2_regularizer(5e-04))

        x = tf.reshape(x, shape=[cfg.batch_size, -1])
        reconstruction_loss = tf.reduce_mean(tf.square(pose_out - x))

    if cfg.weight_reg:
        # regularization loss
        regularization = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # loss+0.0005*reconstruction_loss+regularization#
        loss_all = tf.add_n([loss] + [0.0005 * data_size* data_size * reconstruction_loss] + regularization)
    else:
        loss_all = tf.add_n([loss] + [0.0005 * data_size* data_size * reconstruction_loss])

    return loss_all, loss, reconstruction_loss, pose_out




def kernel_tile(input, kernel, stride):
    # output = tf.extract_image_patches(input, ksizes=[1, kernel, kernel, 1], strides=[1, stride, stride, 1], rates=[1, 1, 1, 1], padding='VALID')
    # input should be a tensor with size as [batch_size, height, width, channels]
    # prepare the input pose matrices and the activations to the correct spatial dimension for voting and EM-routing.
    # tile_filter is like a one-hot vector to pick which neighbor to put in d
    # (14,14,32) --> (14,14,3,3,32) with the 3X3 dimension where the extra 3X3 dimension holds all the 9 neighboring 
    #                               points at location (i,j) 
    #                               Eg d[i,j,0,0,:] hold the upper left neighor at spatial location(i,j)
    # once we have d, we can just do a simple dot product with the weight to create the vote
    # 这一部分相当于是tensorflow中准备输入数据的部分
    # 输出：(50,5,5,9,136) 其中9相当于feature_map上3X3的小块，是第四个维度，最后一个137=8*17相当于通道，是第四个维度
    input_shape = input.get_shape()
    tile_filter = np.zeros(shape=[kernel, kernel, input_shape[3],
                                  kernel * kernel], dtype=np.float32)

    # ?? tile_filter meaning
    for i in range(kernel):
        for j in range(kernel):
            tile_filter[i, j, :, i * kernel + j] = 1.0

    tile_filter_op = tf.constant(tile_filter, dtype=tf.float32)
    print('*** depthwise_conv_shape fileter shape', tile_filter_op)
    
    # (1) (3, 3, 136, 9)


    # tf.nn.depthwise_conv2d: input: 'NHWC' or 'NCHW' data formats
    #                         filter:[filter_height, filter_width, in_channels, channel_multiplier]
    #                         return: output_channel: in_channels * channel_multiplier
    
    # input:  [50,12,12,136]
    # filter: [3,3,136,9]
    # output: [50,6,6,136]
    print('*** before depthwise_conv_shape', input) 
    # (1) (50, 12, 12, 136)
    output = tf.nn.depthwise_conv2d(input, tile_filter_op, strides=[1, stride, stride, 1], padding='VALID')
    print('*** after depthwise_conv_shape', output) 
    # (1) (50, 5, 5, 1224); 5 = ceiling((12-3+1)/2)

    output_shape = output.get_shape()
    output = tf.reshape(output, shape=[int(output_shape[0]), int(
        output_shape[1]), int(output_shape[2]), int(input_shape[3]), kernel * kernel])
    # 注： reshape 是首先排列为一个一维向量，然后按照给出的维度赋值
    #      tf.transpose 是直接将维度顺序打乱
    output = tf.transpose(output, perm=[0, 1, 2, 4, 3])
    print('*** kernel tile output shape', output) 
    # (1) (50, 5, 5, 9, 136)
    return output

# input should be a tensor with size as [batch_size, caps_num_i, 16]
def mat_transform(input, caps_num_c, regularizer, tag=False):
    # extracts the transformation matrices parameters as a tensorflow trainable variable w
    # It then multiplies with the "tiled" input pose matirces to generate the votes for parent capsules
    # 输入[1250,72,16]
    print('*** Inside mat_transform input shape', input)
    batch_size = int(input.get_shape()[0])
    caps_num_i = int(input.get_shape()[1])
    output = tf.reshape(input, shape=[batch_size, caps_num_i, 1, 4, 4])
    # the output of capsule is miu, the mean of a Gaussian, and activation, the sum of probabilities
    # it has no relationship with the absolute values of w and votes
    # using weights with bigger stddev helps numerical stability
    # Note: 这里的w是transformation invariant matrix, 由于不同的capsule层有不同的name, 得以区分
    w = slim.variable('w', shape=[1, caps_num_i, caps_num_c, 4, 4], dtype=tf.float32,
                      initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0),
                      regularizer=regularizer)
    
    # create a new tensor by replication input miltiple times
    # ouput[i] dimension: input.dims(i) * multiples[i] elements
    # 在第一个维度上进行了batch_size次重复，并没有重新赋予其他值

    # 这里的w在BP时候只更新一个w,这里进行tile的复制是为了后面计算方便？？
    
    w = tf.tile(w, [batch_size, 1, 1, 1, 1])
    output = tf.tile(output, [1, 1, caps_num_c, 1, 1])
    print('*** Inside mat_transform w shape', w)          # (1250, 72, 16, 4, 4)
    print('*** Inside mat_trasform output shape', output) # (1250, 72, 16, 4, 4)
    print('*** Inside mat_trasform output tf.matmul(output, w)', tf.matmul(output, w)) # (1250, 72, 16, 4, 4) 
    # (1250, 72, 16, 4, 4) 与 (1250, 72, 16, 4, 4) 相乘，其实是最后两个维度的(4,4)进行矩阵相乘 
    votes = tf.reshape(tf.matmul(output, w), [batch_size, caps_num_i, caps_num_c, 16])
    return votes


def build_arch_baseline(input, is_train: bool, num_classes: int):

    bias_initializer = tf.truncated_normal_initializer(
        mean=0.0, stddev=0.01)  # tf.constant_initializer(0.0)
    # The paper didnot mention any regularization, a common l2 regularizer to weights is added here
    weights_regularizer = tf.contrib.layers.l2_regularizer(5e-04)

    tf.logging.info('input shape: {}'.format(input.get_shape()))

    # weights_initializer=initializer,
    with slim.arg_scope([slim.conv2d, slim.fully_connected], trainable=is_train, biases_initializer=bias_initializer, weights_regularizer=weights_regularizer):
        with tf.variable_scope('relu_conv1') as scope:
            output = slim.conv2d(input, num_outputs=32, kernel_size=[
                                 5, 5], stride=1, padding='SAME', scope=scope, activation_fn=tf.nn.relu)
            output = slim.max_pool2d(output, [2, 2], scope='max_2d_layer1')

            tf.logging.info('output shape: {}'.format(output.get_shape()))

        with tf.variable_scope('relu_conv2') as scope:
            output = slim.conv2d(output, num_outputs=64, kernel_size=[
                                 5, 5], stride=1, padding='SAME', scope=scope, activation_fn=tf.nn.relu)
            output = slim.max_pool2d(output, [2, 2], scope='max_2d_layer2')

            tf.logging.info('output shape: {}'.format(output.get_shape()))

        output = slim.flatten(output)
        output = slim.fully_connected(output, 1024, scope='relu_fc3', activation_fn=tf.nn.relu)
        tf.logging.info('output shape: {}'.format(output.get_shape()))
        output = slim.dropout(output, 0.5, scope='dp')
        output = slim.fully_connected(output, num_classes, scope='final_layer', activation_fn=None)
        tf.logging.info('output shape: {}'.format(output.get_shape()))
        return output


def build_arch(input, coord_add, is_train: bool, num_classes: int):
    test1 = []
    data_size = int(input.get_shape()[1])
    # xavier initialization is necessary here to provide higher stability
    # initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
    # instead of initializing bias with constant 0, a truncated normal initializer is exploited here for higher stability
    bias_initializer = tf.truncated_normal_initializer(
        mean=0.0, stddev=0.01)  # tf.constant_initializer(0.0)
    # The paper didnot mention any regularization, a common l2 regularizer to weights is added here
    weights_regularizer = tf.contrib.layers.l2_regularizer(5e-04)

    tf.logging.info('input shape: {}'.format(input.get_shape()))

    # weights_initializer=initializer,
    with slim.arg_scope([slim.conv2d], trainable=is_train, biases_initializer=bias_initializer, weights_regularizer=weights_regularizer):
        with tf.variable_scope('relu_conv1') as scope:
            # Question: padding should be SAME
            output = slim.conv2d(input, num_outputs=cfg.A, kernel_size=[
                                 5, 5], stride=2, padding='VALID', scope=scope, activation_fn=tf.nn.relu)
            # sirius: padding valid: ceiling[(width - filter_size + 1)/stride]
            data_size = int(np.floor((data_size - 4) / 2))
            
            print('*** after relu_conv1 data size',cfg.batch_size, data_size, data_size, cfg.A)
            assert output.get_shape() == [cfg.batch_size, data_size, data_size, cfg.A]
            tf.logging.info('conv1 output shape: {}'.format(output.get_shape())) # (50, 12, 12, 32)
    
        with tf.variable_scope('primary_caps') as scope:
            # Question: padding should be SAME
            pose = slim.conv2d(output, num_outputs=cfg.B * 16,
                               kernel_size=[1, 1], stride=1, padding='VALID', scope=scope, activation_fn=None)
            print('*** primary_caps, pose before reshape', pose)  # (50, 12, 12, 128)
             
            # Question: activation_fn: tf.nn.sigmoid
            # Question: why not using logistic(lambda()) to compute??
            activation = slim.conv2d(output, num_outputs=cfg.B, kernel_size=[
                                     1, 1], stride=1, padding='VALID', scope='primary_caps/activation', activation_fn=tf.nn.sigmoid)
            print('*** primary_caps, activation before reshape', activation) # (50, 12, 12, 8)

            pose = tf.reshape(pose, shape=[cfg.batch_size, data_size, data_size, cfg.B, 16])
            print('*** primary_caps, pose', pose) # (50, 12, 12, 8, 16)

            activation = tf.reshape(
                activation, shape=[cfg.batch_size, data_size, data_size, cfg.B, 1])
            print('*** primary_caps, activation', activation) # (50, 12, 12, 8, 1)

            output = tf.concat([pose, activation], axis=4)
            output = tf.reshape(output, shape=[cfg.batch_size, data_size, data_size, -1])
            assert output.get_shape() == [cfg.batch_size, data_size, data_size, cfg.B * 17]
            tf.logging.info('primary capsule output shape: {}'.format(output.get_shape())) # (50, 12, 12, 136)

        with tf.variable_scope('conv_caps1') as scope:
            output = kernel_tile(output, 3, 2)
            print('*** Inside conv_caps1, after kernel_tile output', output) # (50, 5, 5, 9, 136)

            data_size = int(np.floor((data_size - 2) / 2)) # 这个变量是专门用来计算feature map的
            print('*** Inside conv_caps1, data_size', data_size) # 5
            
            output = tf.reshape(output, shape=[cfg.batch_size *data_size * data_size, 3 * 3 * cfg.B, 17])
            print('*** Inside conv_caps1, after reshape output', output) # (1250, 72, 17)

            activation = tf.reshape(output[:, :, 16], shape=[cfg.batch_size * data_size * data_size, 3 * 3 * cfg.B, 1])
            print('*** Inside conv_caps1, after reshape activation', activation) # (1250, 72, 1)

            with tf.variable_scope('v') as scope:
                votes = mat_transform(output[:, :, :16], cfg.C, weights_regularizer, tag=True)
                tf.logging.info('conv cap 1 votes shape: {}'.format(votes.get_shape())) # (1250, 72, 16, 16)

            with tf.variable_scope('routing') as scope:
                # votes:       [1250,72,72,16]
                # activations: [1250,72,1]
                miu, activation, _ = em_routing(votes, activation, cfg.C, weights_regularizer)
                tf.logging.info('conv cap 1 miu shape: {}'.format(miu.get_shape())) # (1250, 1, 16, 16)
                tf.logging.info('conv cap 1 activation before reshape: {}'.format(
                    activation.get_shape())) # (1250, 16)

            pose = tf.reshape(miu, shape=[cfg.batch_size, data_size, data_size, cfg.C, 16])
            tf.logging.info('conv cap 1 pose shape: {}'.format(pose.get_shape())) # (50, 5, 5, 16, 16)
            
            activation = tf.reshape(activation, shape=[cfg.batch_size, data_size, data_size, cfg.C, 1])
            tf.logging.info('conv cap 1 activation after reshape: {}'.format(activation.get_shape())) # (50, 5, 5, 16, 1)
            
            output = tf.reshape(tf.concat([pose, activation], axis=4), [cfg.batch_size, data_size, data_size, -1])
            tf.logging.info('conv cap 1 output shape: {}'.format(output.get_shape())) # (50, 5, 5, 272)

        with tf.variable_scope('conv_caps2') as scope:
            output = kernel_tile(output, 3, 1)
            data_size = int(np.floor((data_size - 2) / 1))
            output = tf.reshape(output, shape=[cfg.batch_size *data_size * data_size, 3 * 3 * cfg.C, 17])
            activation = tf.reshape(output[:, :, 16], shape=[cfg.batch_size * data_size * data_size, 3 * 3 * cfg.C, 1])

            with tf.variable_scope('v') as scope:
                votes = mat_transform(output[:, :, :16], cfg.D, weights_regularizer)
                tf.logging.info('conv cap 2 votes shape: {}'.format(votes.get_shape())) # (450, 144, 16, 16)
 
            with tf.variable_scope('routing') as scope:
                miu, activation, _ = em_routing(votes, activation, cfg.D, weights_regularizer)

            pose = tf.reshape(miu, shape=[cfg.batch_size * data_size * data_size, cfg.D, 16]) # (450, 144, 16, 16)
            tf.logging.info('conv cap 2 pose shape: {}'.format(votes.get_shape()))
            activation = tf.reshape(activation, shape=[cfg.batch_size * data_size * data_size, cfg.D, 1])
            tf.logging.info('conv cap 2 activation shape: {}'.format(activation.get_shape())) # (450, 16, 1)

        # It is not clear from the paper that ConvCaps2 is full connected to Class Capsules, or is conv connected with kernel size of 1*1 and a global average pooling.
        # From the description in Figure 1 of the paper and the amount of parameters (310k in the paper and 316,853 in fact), I assume a conv cap plus a golbal average pooling is the design.
        with tf.variable_scope('class_caps') as scope:
            with tf.variable_scope('v') as scope:
                votes = mat_transform(pose, num_classes, weights_regularizer)

                assert votes.get_shape() == [cfg.batch_size * data_size *data_size, cfg.D, num_classes, 16]
                tf.logging.info('class cap votes original shape: {}'.format(votes.get_shape())) # (450, 16, 10, 16)

                coord_add = np.reshape(coord_add, newshape=[data_size * data_size, 1, 1, 2])
                coord_add = np.tile(coord_add, [cfg.batch_size, cfg.D, num_classes, 1])
                coord_add_op = tf.constant(coord_add, dtype=tf.float32)

                votes = tf.concat([coord_add_op, votes], axis=3)
                tf.logging.info('class cap votes coord add shape: {}'.format(votes.get_shape())) # (450, 16, 10, 18)

            with tf.variable_scope('routing') as scope:
                miu, activation, test2 = em_routing(votes, activation, num_classes, weights_regularizer)
                tf.logging.info('class cap activation shape: {}'.format(activation.get_shape())) # (450, 10)
                tf.summary.histogram(name="class_cap_routing_hist",values=test2)

            output = tf.reshape(activation, shape=[cfg.batch_size, data_size, data_size, num_classes])

        output = tf.reshape(tf.nn.avg_pool(output, ksize=[1, data_size, data_size, 1], strides=[
                            1, 1, 1, 1], padding='VALID'), shape=[cfg.batch_size, num_classes])
        tf.logging.info('class cap output shape: {}'.format(output.get_shape())) # (50, 10)
        pose = tf.nn.avg_pool(tf.reshape(miu, shape=[cfg.batch_size, data_size, data_size, -1]), ksize=[
                              1, data_size, data_size, 1], strides=[1, 1, 1, 1], padding='VALID')
        pose_out = tf.reshape(pose, shape=[cfg.batch_size, num_classes, 18])
    return output, pose_out


def test_accuracy(logits, labels):
    logits_idx = tf.to_int32(tf.argmax(logits, axis=1))
    logits_idx = tf.reshape(logits_idx, shape=(cfg.batch_size,))
    correct_preds = tf.equal(tf.to_int32(labels), logits_idx)
    accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32)) / cfg.batch_size

    return accuracy

# The main purpose of the EM routing is to compute the pose matrices and the activations of the output capsules
def em_routing(votes, activation, caps_num_c, regularizer, tag=False):
    print('Inside em_routing =================================================================================')
    print('=== votes shape', votes)
    print('=== activation shape', activation)

    test = []

    batch_size = int(votes.get_shape()[0])
    caps_num_i = int(activation.get_shape()[1])
    n_channels = int(votes.get_shape()[-1])

    sigma_square = []
    miu = []
    activation_out = []
    beta_v = slim.variable('beta_v', shape=[caps_num_c, n_channels], dtype=tf.float32,
                           initializer=tf.constant_initializer(0.0),#tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                           regularizer=regularizer)
    print('=== beta_v shape', beta_v) # [16,16]

    beta_a = slim.variable('beta_a', shape=[caps_num_c], dtype=tf.float32,
                           initializer=tf.constant_initializer(0.0),#tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                           regularizer=regularizer)
    print('=== beta_a shape', beta_a) # [16,]
    # votes_in = tf.stop_gradient(votes, name='stop_gradient_votes')
    # activation_in = tf.stop_gradient(activation, name='stop_gradient_activation')
    
    votes_in = votes
    activation_in = activation

    for iters in range(cfg.iter_routing):
        # if iters == cfg.iter_routing-1:
        # e-step
        if iters == 0:
            r = tf.constant(np.ones([batch_size, caps_num_i, caps_num_c], dtype=np.float32) / caps_num_c)
            print('=== e-step r',r)
            # (iter0) [1250,72,16]
        else:
            # Contributor: Yunzhi Shi
            # log and exp here provide higher numerical stability especially for bigger number of iterations
            log_p_c_h = -tf.log(tf.sqrt(sigma_square)) - \
                        (tf.square(votes_in - miu) / (2 * sigma_square))
            print('=== e-step log_p_c_h_1',log_p_c_h) # [1250,72,16,16]

            # max of a across dimensions in a tensor
            log_p_c_h = log_p_c_h - \
                        (tf.reduce_max(log_p_c_h, axis=[2, 3], keep_dims=True) - tf.log(10.0))
            print('=== e-step log_p_c_h_2',log_p_c_h) # [1250,72,16,16]
            
            p_c = tf.exp(tf.reduce_sum(log_p_c_h, axis=3)) #[1250,72,16]
            print('=== e-step p_c',p_c)

            ap = p_c * tf.reshape(activation_out, shape=[batch_size, 1, caps_num_c]) #[1250,72,16]
            print('=== e-step ap',ap)
            # ap = tf.reshape(activation_out, shape=[batch_size, 1, caps_num_c])

            r = ap / (tf.reduce_sum(ap, axis=2, keep_dims=True) + cfg.epsilon) #[1250,72,16]
            print('=== e-step final r',r)


        # m-step
        print('^^^ m-step r_0',r)
        print('^^^ m-step activation_in',activation_in)
        r = r * activation_in
        # (iter0): [1250,72,16] * [1250,72,1] --> [1250,72,16]
        print('^^^ m-step r_1',r)
        
        r = r / (tf.reduce_sum(r, axis=2, keep_dims=True)+cfg.epsilon)
        # (iter0): [1250,72,16]
        print('^^^ m-step r_2',r)

        r_sum = tf.reduce_sum(r, axis=1, keep_dims=True)
        # (iter0): [1250,1,16]
        print('^^^ m-step r_sum',r_sum)

        r1 = tf.reshape(r / (r_sum + cfg.epsilon),
                        shape=[batch_size, caps_num_i, caps_num_c, 1]) 
        print('^^^ m-step r1',r1) # [1250,72,16,1]


        miu = tf.reduce_sum(votes_in * r1, axis=1, keep_dims=True) # [1250,1,16,16]
        print('^^^ m-step miu',miu)

        sigma_square = tf.reduce_sum(tf.square(votes_in - miu) * r1,
                                     axis=1, keep_dims=True) + cfg.epsilon # [1250,1,16,16]
        print('^^^ m-step sigma_square',sigma_square)


        if iters == cfg.iter_routing-1:
            r_sum = tf.reshape(r_sum, [batch_size, caps_num_c, 1])
            print('^^^ m-step iter r_sum',r_sum)
            
            cost_h = (beta_v + tf.log(tf.sqrt(tf.reshape(sigma_square,
                                                         shape=[batch_size, caps_num_c, n_channels])))) * r_sum
            print('^^^ m-step iter cost_h',cost_h)

            activation_out = tf.nn.softmax(cfg.ac_lambda0 * (beta_a - tf.reduce_sum(cost_h, axis=2)))
            print('^^^ m-step iter activation_out',activation_out)
        else:
            # 师姐说这部分可能没用
            activation_out = tf.nn.softmax(r_sum)
            print('^^^ m-step iter else activation_out',activation_out) # [1250,1,16]

        # if iters <= cfg.iter_routing-1:
        #     activation_out = tf.stop_gradient(activation_out, name='stop_gradient_activation')

    return miu, activation_out, test
