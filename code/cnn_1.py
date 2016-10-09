# -*-coding:utf-8 -*-


#https://www.tensorflow.org/versions/r0.11/tutorials/mnist/pros/index.html
import tensorflow as tf 




def _initialize_var_on_cpu(shape,initializer,name='weight_variable'):
	'''Helper function  to create a Variable on cpu memory
	Args:
		name:name of variable
		initializer for Variable
	'''
	with tf.device('/cpu:0'):
		var=tf.get_variable(shape=shape,initializer=initializer,name=name)
	return var


def weight_variable(shape,name='weight_variable'):
	'''
		Instead of doing this repeatedly while we build the model, let's create two handy functions to do it for us.
	'''
	print tf.app.flags.FLAGS.stddev
	var=_initialize_var_on_cpu(shape,
		tf.truncated_normal_initializer(tf.app.flags.FLAGS.stddev,dtype=tf.float32),name=name)
	return var

def bias_variable(shape):
	init_bias=tf.constant(0.1,shape=shape)
	return tf.Variable(init_bias)


def conv2d(x,W):
	'''
		this convolutions use a stride of one and are zero padded so that the output is the same as the input after convlutional ooperation.
	Arg:
		x:  input the convolution operation
		W: weight


	'''
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')



def cnn_model(x,keep_prob):
	'''
	Note:
		As paper of VGGNet (2014 CVPR)proposed that ,it will be good to use small kernel size ,for example:3x3.
	GoogleNet(CVPR 2014,2015,2016) points out : for example,we can use two 3*1  with two layers kernel instead 
	of 3*1 with just one layer ,which can reduce the parametes from 9 to 6 with more sophisticated network structure .
	Another question is how to add or sub the layer is core problem of me ???
	Train and test with differnt probabitly of dropout, this means that if you want get the accuracy in training time,
	you must set the probabilty with 1.0 and in traing you can set any float number between 0.0-1.0 ,default is 0.5.


	'''
	#define the first layer
	with tf.variable_scope('conv1') as scope:
		w_conv1=weight_variable([5,5,1,32],name='conv1') # convolutional kernel is 5*5 and the input channel is 1 with the output channel of 32
		b_conv1=bias_variable([32])

		# reshpe the image to a 4d tensor ,with the sencond and third corrponding the length and width of image and the final dimension as the channels
		x_image=tf.reshape(x,[-1,28,28,1])

		#h_conv1=tf.nn.relu(conv2d(x_image,w_conv1)+b_conv1)
		h_conv1=tf.nn.relu(tf.nn.bias_add(conv2d(x_image,w_conv1),b_conv1))
		h_pool1=max_pool_2x2(h_conv1)

	# define the second layer of cnn
	with tf.variable_scope('conv2') as scope:
		w_conv2=weight_variable([5,5,32,64],name='conv2')
		b_conv2=bias_variable([64])
		h_conv2=tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2)
		h_pool2=max_pool_2x2(h_conv2)

	# define the fully connected layer with dropout for reducing overfitting
	with tf.variable_scope('fullyc1') as scope:
		w_fc1=weight_variable([7*7*64,1024],name='fc1')
		b_fc1=bias_variable([1024])
		h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
		h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)

		
		h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob=keep_prob,seed=tf.app.flags.FLAGS.seed)


	#define the second fully connected layer without dropout

	#define the third fully connected layer for output

	w_fc3=weight_variable([1024,10],name='fc3')
	b_fc3=bias_variable([10])
	y_conv=tf.matmul(h_fc1,w_fc3)+b_fc3

	return y_conv


