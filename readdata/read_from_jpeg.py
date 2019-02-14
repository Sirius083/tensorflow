
import tensorflow as tf
# read jpeg file
filename = 'E:/tiny_imagenet/tiny-imagenet-200/train\n01443537\images\n01443537_0.JPEG'
image_string = tf.read_file(filename)
image_decoded = tf.image.decode_jpeg(image_string)
image_decoded.set_shape([64, 64, 3])

# plot from tf.tensor
with tf.Session() as sess:
    img = sess.run(image_decoded)   
import matplotlib.pyplot as plt
# image_path = 'E:/tiny_imagenet/tiny-imagenet-200/train\\n01443537\\images\\n01443537_0.JPEG'
# img = plt.imread(image_path)
imgplot = plt.imshow(img)
plt.show()
