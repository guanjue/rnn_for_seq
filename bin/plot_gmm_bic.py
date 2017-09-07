import os,sys
from tensorflow.python.ops import rnn
import tensorflow as tf
import numpy as np
import math
import random
from sklearn import preprocessing
from sklearn.decomposition import PCA, KernelPCA
from sklearn import svm
import numpy as np
from rdp import rdp
import matplotlib.pyplot as plt
from scipy import signal
from sklearn import mixture
import matplotlib.mlab as mlb
from scipy import stats
import itertools
from sklearn.cluster import KMeans

################################################################################################
def read2d_array(filename):
	r1=open(output,'w')
	for records in array:
		for i in range(0,len(records)-1):
			r1.write(str(records[i])+'\t')
		r1.write(str(records[len(records)-1])+'\n')
	r1.close()

def write2d_array(array,output):
	r1=open(output,'w')
	for records in array:
		for i in range(0,len(records)-1):
			r1.write(str(records[i])+'\t')
		r1.write(str(records[len(records)-1])+'\n')
	r1.close()
################################################################################################
### read np array hidden layer predict
data0_npz = np.load('rnn_hidden_layer_pos_pred_transpose.npz')
data0 = []
for i in data0_npz:
	data0.append(data0_npz[i])
data0 = np.array(data0)
data0 = data0[0]
### read np array hidden layers
rnn_h_npz = np.load('rnn_hidden_layer_reshape.npz')
rnn_h_npz0 = []
for i in rnn_h_npz:
	rnn_h_npz0.append(rnn_h_npz[i])
rnn_h_npz0 = np.array(rnn_h_npz0)
rnn_h_npz0 = rnn_h_npz0[0]
print('rnn_h_npz0.shape')
print(rnn_h_npz0.shape)
################################################################################################
### sampling from all hidden states
np.random.seed(seed=2017)
rsample_id = np.random.randint(data0.shape[0]-1, size=10000)
data0_sample = data0[rsample_id,:]
data0_sample = data0_sample.flatten()
print('data0_sample.shape')
print(data0_sample.shape)
################################################################################################
print('Start GMM!')
histdist = plt.hist(data0_sample, 1000, normed=True, histtype='bar',color='k')
data0_sample = data0_sample.reshape(data0_sample.shape[0],1)
lowest_bic = 100000000000
bic=[]
for i in range(1,11):
	gmm = mixture.GaussianMixture(n_components=i, covariance_type='full', max_iter=100, tol=0.001, random_state=2017).fit(data0_sample)
	bic.append(gmm.bic(data0_sample))
	if bic[-1] < lowest_bic:
		lowest_bic = bic[-1]
		best_gmm = gmm
		print('lowest_bic: '+str(i)+' components')
		print(lowest_bic)
### plot gmm BIC
xpos=np.array(range(1,len(bic)+1))
bic = np.array(bic)
plt.scatter(xpos, bic)
plt.xlim( (0, bic.shape[0]+2) )
plt.ylim( (np.min(bic)-10000, np.max(bic)+10000) )
plt.ylabel('BIC')
plt.xlabel('number of component in GMM')
plt.savefig('gmm_bic.pdf')
plt.close()
#gmm = mixture.GaussianMixture(n_components=10, covariance_type='full', max_iter=1000, tol=0.000001).fit(data0_sample.reshape(data0_sample.shape[0],1))
print('GMM Done!')






