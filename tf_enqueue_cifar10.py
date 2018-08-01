# code from https://github.com/adventuresinML/adventures-in-ml-code/blob/master/tf_queuing.py

def cifar_shuffle_batch():
    batch_size = 128
    num_threads = 16
    # create a list of all our filenames
    filename_list = [data_path + 'data_batch_{}.bin'.format(i + 1) for i in range(5)]
    # create a filename queue
    # file_q = cifar_filename_queue(filename_list)
    file_q = tf.train.string_input_producer(filename_list)
    # read the data - this contains a FixedLengthRecordReader object which handles the
    # de-queueing of the files.  It returns a processed image and label, with shapes
    # ready for a convolutional neural network
    image, label = read_data(file_q)
    # setup minimum number of examples that can remain in the queue after dequeuing before blocking
    # occurs (i.e. enqueuing is forced) - the higher the number the better the mixing but
    # longer initial load time
    min_after_dequeue = 10000
    # setup the capacity of the queue - this is based on recommendations by TensorFlow to ensure
    # good mixing
    capacity = min_after_dequeue + (num_threads + 1) * batch_size
    # image_batch, label_batch = cifar_shuffle_queue_batch(image, label, batch_size, num_threads)
    image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size, capacity, min_after_dequeue,
                                                      num_threads=num_threads)
    # now run the training
    cifar_run(image_batch, label_batch)


def cifar_run(image, label):
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(5):
            image_batch, label_batch = sess.run([image, label])
            print(image_batch.shape, label_batch.shape)

        coord.request_stop()
        coord.join(threads)


def cifar_filename_queue(filename_list):
    # convert the list to a tensor
    string_tensor = tf.convert_to_tensor(filename_list, dtype=tf.string)
    # randomize the tensor
    tf.random_shuffle(string_tensor)
    # create the queue
    fq = tf.FIFOQueue(capacity=10, dtypes=tf.string)
    # create our enqueue_op for this q
    fq_enqueue_op = fq.enqueue_many([string_tensor])
    # create a QueueRunner and add to queue runner list
    # we only need one thread for this simple queue
    tf.train.add_queue_runner(tf.train.QueueRunner(fq, [fq_enqueue_op] * 1))
    return fq


def cifar_shuffle_queue_batch(image, label, batch_size, capacity, min_after_dequeue, threads):
    tensor_list = [image, label]
    dtypes = [tf.float32, tf.int32]
    shapes = [image.get_shape(), label.get_shape()]
    q = tf.RandomShuffleQueue(capacity=capacity, min_after_dequeue=min_after_dequeue,
                              dtypes=dtypes, shapes=shapes)
    enqueue_op = q.enqueue(tensor_list)
    # add to the queue runner
    tf.train.add_queue_runner(tf.train.QueueRunner(q, [enqueue_op] * threads))
    # now extract the batch
    image_batch, label_batch = q.dequeue_many(batch_size)
    return image_batch, label_batch
