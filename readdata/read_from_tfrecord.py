# tensorflow 2018 summit dataset
import tensorflow as tf
tf.contrib.data.prefetch_to_device() # GPU

file_names = [f[0] for f in filenames_labels]
files = tf.data.Dataset.list_files(file_pattern)


file_pattern = glob.glob('E:/tiny_imagenet/tiny-imagenet-200/train/*/images/*.JPEG')

file_pattern = r'E:/tiny_imagenet/tiny-imagenet-200/train/*/images/*.JPEG'

files = tf.data.Dataset.list_files(file_pattern)
dataset = tf.data.TFRecordDataset(files, num_parallel_reads = 8)

dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(10000, NUM_EPOCHS))
dataset = dataset.apply(tf.contrib.data.map_and_batch x:..., BATCH_SIZE)

iterator = dataset.make_one_shot_iterator()
features = iterator.get_next()

