# 1. sys.path.append(official upper directory)
# 2. download cifar10 binary version, change data_dir in get_filenames
# 3. Time: add train_accuracy_perbatch after defination of accuracy 
#          in resnet_model_fn() in resnet_run_loop.py
# 4. models-master/official/utils/logs/hooks_helper.py line 33 change to
# _TENSORS_TO_LOG = dict((x, x) for x in ['learning_rate',
#                                         'cross_entropy',
#                                         'train_accuracy',
#                                         'train_accuracy_perbatch'])
