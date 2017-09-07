import os,sys
from tensorflow.python.ops import rnn
import tensorflow as tf
import numpy as np
import math
import random
from sklearn import preprocessing
from sklearn.decomposition import PCA, KernelPCA
from sklearn import svm

data_test0 = np.load('test.npz')
data_test = []
for i in data_test0:
	data_test.append(data_test0[i])
xs_test = data_test[1]
ys_test = data_test[2]

xs_test_matrix = []
for sequence in xs_test:
	tmp_matrix = []
	for i in sequence:
		tmp_str = [[0,0,0,0]]
		tmp_str[0][i] = 1
		tmp_matrix.append(tmp_str)
	xs_test_matrix.append(tmp_matrix)
xs_test_matrix = np.array(xs_test_matrix)
print('xs_test_matrix.shape')
print(xs_test_matrix.shape)

ys_test_matrix = []
for label in ys_test:
	if label == 0:
		ys_test_matrix.append([0.0,1.0])
	else:
		ys_test_matrix.append([1.0,0.0])
ys_test_matrix = np.array(ys_test_matrix)
print('ys_test_matrix.shape')
print(ys_test_matrix.shape)


data_train0 = np.load('train.npz')
data_train = []
for i in data_train0:
	data_train.append(data_train0[i])
xs_train = data_train[1]
ys_train = data_train[2]

xs_train_matrix = []
for sequence in xs_train:
	tmp_matrix = []
	for i in sequence:
		tmp_str = [[0,0,0,0]]
		tmp_str[0][i] = 1
		tmp_matrix.append(tmp_str)
	xs_train_matrix.append(tmp_matrix)
xs_train_matrix = np.array(xs_train_matrix)
print('xs_train_matrix.shape')
print(xs_train_matrix.shape)

ys_train_matrix = []
for label in ys_train:
	if label == 0:
		ys_train_matrix.append([0.0,1.0])
	else:
		ys_train_matrix.append([1.0,0.0])
ys_train_matrix = np.array(ys_train_matrix)
print('ys_train_matrix.shape')
print(ys_train_matrix.shape)

sec_d=100 
thr_d=1
for_d=4

filter1_size1=24
filter1_size2=1
filter1_size_out=32

filter1_max_pool_size=100

full_cn_out = 64

iter_num=10000
batch_size=100
training_speed=0.0001



def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.01)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.01, shape=shape)
	return tf.Variable(initial)

### Convolution and Pooling
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def conv2d_1(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_n(x, max_pool_size):
	return tf.nn.max_pool(x, ksize=[1, max_pool_size, max_pool_size, 1], strides=[1, max_pool_size, max_pool_size, 1], padding='SAME')

######### Tensorflow model
x = tf.placeholder(tf.float32, shape=[None, sec_d, thr_d, for_d])
y_ = tf.placeholder(tf.float32, shape=[None, 2])

### First Convolutional Layer
### The First layer will have 32 features for each 6x1 patch.
W_conv1 = weight_variable([filter1_size1, filter1_size2, for_d, filter1_size_out])
b_conv1 = bias_variable([filter1_size_out])
### 
pool_shape1=sec_d*thr_d/filter1_max_pool_size
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, filter1_max_pool_size, 1, 1],strides=[1, filter1_max_pool_size, 1, 1], padding='SAME')

### Densely Connected Layer
### a fully-connected layer with 1024 neurons to allow processing on the entire seq. 
W_fc1 = weight_variable([pool_shape1 * filter1_size_out, full_cn_out])
b_fc1 = bias_variable([full_cn_out])

###
h_pool_flat = tf.reshape(h_pool1, [-1, pool_shape1 * filter1_size_out])
h_fc1 = (tf.matmul(h_pool_flat, W_fc1) + b_fc1)

### Dropout
### To reduce overfitting, we will apply dropout before the readout layer. 
### We create a placeholder for the probability that a neuron's output is kept during dropout
keep_prob1 = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob1)

### Readout Layer
### Finally, we add a softmax layer, just like for the one layer softmax regression above.
W_fc2 = weight_variable([full_cn_out, ys_train_matrix.shape[1]])
b_fc2 = bias_variable([ys_train_matrix.shape[1]])
### 
y_conv=(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
y_conv_pred = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
### Evaluate the Model
#cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_) 
train_step = tf.train.AdamOptimizer(training_speed).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
sess.run(tf.initialize_local_variables())




print('Start!!! LSTM')
#saver = tf.train.Saver()
#saver.restore(sess, "trained_cnn_model.ckpt")
k=0
index_array_p=np.arange(ys_train.shape[0])
np.random.shuffle(index_array_p)

for i in range(iter_num):
	if (k+1)*batch_size > ys_train.shape[0]:
		k=0
		index_array_p=np.arange(ys_train.shape[0])
		np.random.shuffle(index_array_p)

	batch_id = index_array_p[k*batch_size:(k+1)*batch_size]
	batch_xs1=xs_train_matrix[batch_id,:,:,:]
	batch_ys1=ys_train_matrix[batch_id,:]
	k=k+1
	#print(batch_xs1.shape)
	#print(batch_xs1_2.shape)
	#print(batch_xs1_3.shape)
	sess.run(train_step, feed_dict={x: batch_xs1, y_: batch_ys1, keep_prob1: 0.5})

	if i%100 == 0:
		train_accuracy = accuracy.eval(feed_dict={x:batch_xs1, y_: batch_ys1, keep_prob1: 1.0})
		print("step %d, training accuracy same cell!!! R2: %g"%(i, train_accuracy) )

	#if i%500 == 0:
	#	cost_v = sess.run(cost, feed_dict={x: test_x[0:int(test_y.shape[0]/10)], y_: test_y[0:int(test_y.shape[0]/10)], keep_prob1: 1})
	#	print("step %d, TESTING accuracy same cell!!!: %g"%(i, cost_v/int(test_y.shape[0]/10)*batch_size) )
	if i%1000 == 0:
		test_accuracy = accuracy.eval(feed_dict={x:xs_test_matrix, y_: ys_test_matrix, keep_prob1: 1.0})
		print("step %d, testing accuracy same cell!!! R2: %g"%(i, test_accuracy) )

		if 1==1:#accuracy_r2_test<=r2_test and accuracy_r2_train<=r2_train:
			saver = tf.train.Saver()
			save_path = saver.save(sess, "trained_cnn_model.ckpt")

test_accuracy = accuracy.eval(feed_dict={x:xs_test_matrix, y_: ys_test_matrix, keep_prob1: 1.0})
print("step %d, testing accuracy same cell!!! R2: %g"%(i, test_accuracy) )
