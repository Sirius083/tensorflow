# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 10:46:45 2018

@author: Sirius

run train and test at same session
stanford tensorflow deep learning course
https://docs.google.com/presentation/d/1lmcQVNAmJrL8x3Iq0VB1mVaka1r6pOIb-TMVTX5Rufc/edit#slide=id.g2f00a43678_0_268
"""

# iterator
iterator = dataset.make_one_shot_iterator()       # iterates through the dataset exactly once, no need to initialization
iterator = dataset.make_initializable_iterator()  # iterates through the dataset as many times as we want

# 
iterator = dataset.make_one_shot_iterator()
X,Y = iterator.get_next()
with tf.Session() as sess:
	print(sess.run([X,Y]))


iterator = dataset.make_initializable_iterator() 
for i in range(100):
	sess.run(iterator.initializer)
	total_los = 0
	try: 
		while True:
			sess.run([optimizer])
		except tf.errors.OutOfRangeError:
			pass

dataset = dataset.shuffle(1000)
dataset = dataset.repeat(100)
dataset = dataset.batch(128)
dataset = dataset.map(lambda x: tf.one_hot(x,10))

# mnist example
train_data = tf.data.Dataset.from_tensor_slices(train)
train_data = train_data.shuffle(10000)
test_data = tf.data.Dataset.from_tensor_slices(test)

iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
img, laber = iterator.get_next()

train_init = iterator.make_initializer(train_data)
test_init = iterator.make_initializer(test_data)

with tf.Session() as sess:
	for i in range(n_epochs):
		sess.run(train_init)
		try:
			while True:
				_,l = sess.run([optimizer, loss])

		except tf.errors.OutOfRangeError:
			pass
	
	# test the model
	sess.run(test_init)
	try:
		while True
		      sess.run(accuracy)
	except tf.errors.OutOfRangeError:
		pass

    








