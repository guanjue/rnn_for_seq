import os,sys
from tensorflow.python.ops import rnn
import tensorflow as tf
import numpy as np
import math
import random
from sklearn import preprocessing
from sklearn.decomposition import PCA, KernelPCA
from sklearn import svm


def write2d_array(array,output):
	r1=open(output,'w')
	for records in array:
		for i in range(0,len(records)-1):
			r1.write(str(records[i])+'\t')
		r1.write(str(records[len(records)-1])+'\n')
	r1.close()


data_test0 = np.load('test.npz')
xs_test = data_test0['data']
ys_test = data_test0['label']

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
xs_train = data_train0['data']
ys_train = data_train0['label']

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

filter1_size1=16
filter1_size2=1
filter1_size_out=32

filter1_max_pool_size=100

full_cn_out = 64

iter_num=10000
batch_size=100
training_speed=0.001

def bias_constant(shape):
	initial = tf.constant(-1.5, shape=shape)
	return initial

def weight_dimerscan_variable():
	weight_dimerscaner=[]
	### get dimer list
	dimer_list=[]
	for i in range(0,4):
		for j in range(0,4):
			dimer_list.append([i,j])
	### get scanner list
	for ids in dimer_list:
		tmp_matrix=np.zeros([2,1,4])
		tmp_matrix[0,0,ids[0]]=1
		tmp_matrix[1,0,ids[1]]=1
		weight_dimerscaner.append(tmp_matrix)
	### change to np array
	weight_dimerscaner=np.array(weight_dimerscaner)
	weight_dimerscaner=np.transpose(weight_dimerscaner,(1,2,3,0))
	weight_dimerscaner=tf.constant(weight_dimerscaner,dtype=tf.float32)
	print(weight_dimerscaner.shape)
	return weight_dimerscaner

def get_kmer_matrix(k):
	kmer_matrix = np.zeros((pow(4, k),k)) # initial kmer-matrix
	digit = range(4) # get nuc list
	for i in range(kmer_matrix.shape[1]): # for each column in kmer-matrix (each k-mer position)
		n = 0
		m = 0
		for j in range(kmer_matrix.shape[0]): # for each row in kmer-matrix (each k-mer)
			if m%(kmer_matrix.shape[0]/pow(4,i+1)) == 0: 
				if n ==len(digit):
					n=0
				used_digit = digit[n]
				n = n+1
			kmer_matrix[j,i] = used_digit
			m = m+1
	return kmer_matrix

def weight_kmerscan_variable(k):
	weight_kmerscaner=[]
	### get kmer list
	kmer_list=get_kmer_matrix(k)
	#print(kmer_list)
	### get scanner list
	for ids in kmer_list:
		tmp_matrix=np.zeros([k,1,4])
		for p in range(0,k):
			tmp_matrix[p,0,int(ids[p])]=1
		weight_kmerscaner.append(tmp_matrix)
	### change to np array
	weight_kmerscaner=np.array(weight_kmerscaner)
	weight_kmerscaner=np.transpose(weight_kmerscaner,(1,2,3,0))
	weight_kmerscaner=tf.constant(weight_kmerscaner,dtype=tf.float32)
	print(weight_kmerscaner.shape)
	return weight_kmerscaner

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
keep_prob1 = tf.placeholder(tf.float32)
W_conv1 = weight_kmerscan_variable(2)
b_conv1 = bias_constant([filter1_size1])
### 
#pool_shape1=sec_d*thr_d/max_pool1
#h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_conv1 = (tf.nn.relu(conv2d(x, W_conv1) + b_conv1))
print('tf.shape(h_conv1)')
print(tf.shape(h_conv1))
#h_conv1_nonzero = tf.cast(h_conv1==2,h_conv1.dtype)
#print('tf.shape(h_conv1_nonzero)')
#print(tf.shape(h_conv1_nonzero))
x_transpose = tf.transpose(h_conv1, [1, 0, 2, 3])
x_reshape = tf.reshape(x_transpose, [-1, filter1_size1])


# Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
rnn_input = tf.split(x_reshape,axis=0, num_or_size_splits=sec_d)
rnn_cell_num=16
rnn_cell = tf.nn.rnn_cell.GRUCell(rnn_cell_num)

outputs, states = tf.contrib.rnn.static_rnn(rnn_cell, rnn_input, dtype=tf.float32)
w_fc =  weight_variable([rnn_cell_num, 2])
y_conv = tf.matmul(outputs[-1], w_fc )# + bias_variable([2]) )
y_predict = tf.nn.softmax(y_conv)
### 
#y_conv=(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
#y_conv_pred = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
### Evaluate the Model
#cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_) 
train_step = tf.train.AdamOptimizer(training_speed).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

au_roc = tf.metrics.auc(y_, y_predict, curve='ROC')
au_prc = tf.metrics.auc(y_, y_predict, curve='PR')

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
sess.run(tf.initialize_local_variables())




print('Start!!! RNN')
saver = tf.train.Saver()
saver.restore(sess, "trained_rnn_gru_model.ckpt")
k=0

test_accuracy = accuracy.eval(feed_dict={x:xs_test_matrix, y_: ys_test_matrix, keep_prob1: 1.0})
print("testing accuracy same cell!!! R2: "+str(test_accuracy) )

test_roc = sess.run(au_roc, feed_dict={x:xs_test_matrix, y_: ys_test_matrix, keep_prob1: 1.0})
test_prc = sess.run(au_prc, feed_dict={x:xs_test_matrix, y_: ys_test_matrix, keep_prob1: 1.0})
print("testing au roc same cell!!! R2: "+str(test_roc[0]) )
print("testing au prc same cell!!! R2: "+str(test_prc[0]) )

fc_layer = np.array(sess.run(w_fc, feed_dict={x: xs_test_matrix, y_: ys_test_matrix, keep_prob1: 1.0}))
print('fc_layer.shape')
print(fc_layer.shape)

rnn_hidden_layer = np.array(sess.run(outputs, feed_dict={x: xs_test_matrix, y_: ys_test_matrix, keep_prob1: 1.0}))
print('rnn_hidden_layer.shape')
print(rnn_hidden_layer.shape)

for i in range(0,rnn_hidden_layer.shape[2]):
	rnn_hidden_layer_channel = np.transpose(rnn_hidden_layer[:,:,i])
	write2d_array(rnn_hidden_layer_channel,'rnn_hidden_layer_channel/rnn_hidden_layer_channel'+str(i)+'.txt')

rnn_hidden_layer_transpose = np.transpose(rnn_hidden_layer, (1,0,2))
rnn_hidden_layer_reshape = rnn_hidden_layer_transpose.reshape(rnn_hidden_layer_transpose.shape[0]*rnn_hidden_layer_transpose.shape[1], rnn_hidden_layer_transpose.shape[2])
write2d_array(rnn_hidden_layer_reshape,'rnn_hidden_layer_reshape.txt')
np.savez('rnn_hidden_layer_reshape.npz', rnn_hidden_layer_reshape)

np.savez('rnn_hidden_layer.npz', rnn_hidden_layer)

rnn_hidden_layer_pred_flat = np.dot(rnn_hidden_layer_reshape, fc_layer)
write2d_array(rnn_hidden_layer_pred_flat,'rnn_hidden_layer_pred_flat.txt')

rnn_hidden_layer_pred = np.dot(rnn_hidden_layer, fc_layer)
print('rnn_hidden_layer_pred.shape')
print(rnn_hidden_layer_pred.shape)

rnn_hidden_layer_pos_pred_transpose = np.transpose(rnn_hidden_layer_pred[:,:,0])
rnn_hidden_layer_neg_pred_transpose = np.transpose(rnn_hidden_layer_pred[:,:,1])

write2d_array(rnn_hidden_layer_pos_pred_transpose,'rnn_hidden_layer_pos_pred_transpose.txt')
np.savez('rnn_hidden_layer_pos_pred_transpose.npz', rnn_hidden_layer_pos_pred_transpose)

rnn_hidden_layer_pred_softmax = np.exp(rnn_hidden_layer_pos_pred_transpose) / (np.exp(rnn_hidden_layer_pos_pred_transpose) + np.exp(rnn_hidden_layer_neg_pred_transpose))
write2d_array(rnn_hidden_layer_pred_softmax,'rnn_hidden_layer_pred_softmax.txt')

y_predict_matrix = np.array(sess.run(y_predict, feed_dict={x: xs_test_matrix, y_: ys_test_matrix, keep_prob1: 1.0}))
print('y_predict_matrix.shape')
print(y_predict_matrix.shape)
write2d_array(y_predict_matrix,'y_predict_matrix.txt')

y_conv_matrix = np.array(sess.run(y_conv, feed_dict={x: xs_test_matrix, y_: ys_test_matrix, keep_prob1: 1.0}))
print('y_conv_matrix.shape')
print(y_conv_matrix.shape)
write2d_array(y_conv_matrix,'y_conv_matrix.txt')

y_obs_matrix = np.array(sess.run(y_, feed_dict={x: xs_test_matrix, y_: ys_test_matrix, keep_prob1: 1.0}))
print('y_obs_matrix.shape')
print(y_obs_matrix.shape)
write2d_array(y_obs_matrix,'y_obs_matrix.txt')

h_conv1_matrix = np.array(sess.run(h_conv1, feed_dict={x: xs_test_matrix, y_: ys_test_matrix, keep_prob1: 1.0}))
h_conv1_matrix_digit = np.argmax(h_conv1_matrix[:,:,0,:],axis=2)
write2d_array(h_conv1_matrix_digit,'h_conv1_matrix_digit.txt')

x_matrix = np.array(sess.run(x, feed_dict={x: xs_test_matrix, y_: ys_test_matrix, keep_prob1: 1.0}))
x_matrix_digit = np.argmax(x_matrix[:,:,0,:],axis=2)
write2d_array(x_matrix_digit,'x_matrix_digit.txt')


