local_device_protos = device_lib.list_local_devices()
num_gpus = sum([1 for d in local_device_protos if d.device_type == 'GPU'])

# If we are running multi-GPU, we need to wrap the optimizer.
if multi_gpu:
   optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)
