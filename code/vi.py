# -*-coding:utf-8 -*-

import pandas as pd
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt 
import matplotlib.cm as cm

from cnn_1 import cnn_model

tf.app.flags.DEFINE_float('stddev',0.1,'weight variable standard')
tf.app.flags.DEFINE_integer('seed',0,'the random seed to use for reuse')
tf.app.flags.DEFINE_float('learning_rate',1e-2,'learning rate of the model')
tf.app.flags.DEFINE_integer('batch_size',0,'batch size ')

IMAGE_SIZE=28

VALIDATION_SIZE=2000






def display(im_array):
	'''Display an images of numpy array
	Args:
		im_array: images digits strore in numpy array. eg: mnist: shape (784,)
	'''
	image=im_array.reshape(IMAGE_SIZE,IMAGE_SIZE) # grayscale image
	plt.imshow(image,cmap=cm.binary)



def convert_dense_labels(labels,class_number):
	'''coonvert class labels from scalars to one-hot vectors
	Agrs:
		lables: np.array of shape(example_len,)
		class_number: number fo class number
	Return：
		dense_labels: one-hot format vector of class
	'''
	num_labels=len(labels)
	index_offset=np.arange(num_labels)*class_number # example np.arange(3)*4 =array([0, 4, 8])
	dense_labels=np.zeros((num_labels,class_number))
	dense_labels.flat[index_offset+labels]=1    #It allows iterating over the array as if it were a 1-D array

	# print dense_labels # test
	return dense_labels






data=pd.read_csv('../data/train.csv') # total number 42000

images=data.iloc[:,1:].values # np.array()  
images=np.multiply(images,1.0/255) # normalize  the image pixel to be in range (0-1)

labels=data[[0]].values.ravel() # or values.reshape(len,)
labels=convert_dense_labels(labels,10)

# split data into training & validation
validation_images = images[:VALIDATION_SIZE]
validation_labels = labels[:VALIDATION_SIZE]

train_images = images[VALIDATION_SIZE:]
train_labels = labels[VALIDATION_SIZE:]


epoches_completed=0
index_in_epoch=0

num_train_number=train_images.shape[0]

def next_batch(batch_size):

	global train_images
	global train_labels
	global index_in_epoch
	global epoches_completed

	start_index=index_in_epoch
	index_in_epoch+=batch_size
	if index_in_epoch>num_train_number:# finish this example
		epoches_completed+=1
		# shuffle the data again
		permuation=np.arange(num_train_number)
		np.random.shuffle(permuation)
		train_images=train_images[permuation]
		train_labels=train_labels[permuation]

		start_index=0
		index_in_epoch=batch_size
		assert batch_size<num_train_number
	end_index=index_in_epoch
	return train_images[start_index:end_index],train_labels[start_index:end_index]





def train():
	print 'sfsdf'
	x = tf.placeholder(tf.float32, shape=[None, 784])

	y_ = tf.placeholder(tf.float32, shape=[None, 10])



	keep_prob=tf.placeholder(tf.float32)

	y_conv=cnn_model(x,keep_prob)
	cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv,y_))
	train_step=tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
	correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
	accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
	init=tf.initialize_all_variables()
	with tf.Session() as sess:
		sess.run(init)
		for step in range(30000):
			batch=next_batch(300)


			# if step%100==9:
			# 	# print sess.run(batch[1])
			# 	train_accuracy=accuracy.eval(feed_dict={x:train_images,y_:train_labels,keep_prob:1.0})
			# 	print 'step %d ,training accuracy:%g'%(step,train_accuracy)
			train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
			# if step%10
			print("evaluation accuracy %g"%accuracy.eval(feed_dict={
    				x: validation_images, y_: validation_labels, keep_prob: 1.0}))
 
train()






