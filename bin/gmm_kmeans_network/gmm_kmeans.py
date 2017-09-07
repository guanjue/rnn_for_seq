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
data0_sample = data0_sample.reshape(data0_sample.shape[0],1)
#gmm = mixture.BayesianGaussianMixture(n_components=6, covariance_type='full', max_iter=100, tol=0.001, mean_precision_prior=0.001, weight_concentration_prior=0.001, random_state=2017).fit(data0_sample)
#gmm = mixture.GaussianMixture(n_components=10, covariance_type='full', max_iter=1000, tol=0.000001).fit(data0_sample.reshape(data0_sample.shape[0],1))
lowest_bic = 100000000000
bic=[]
for i in range(1,11):
	gmm_tmp = mixture.GaussianMixture(n_components=i, covariance_type='full', max_iter=100, tol=0.001, random_state=2017).fit(data0_sample)
	bic.append(gmm_tmp.bic(data0_sample))
	if bic[-1] < lowest_bic:
		lowest_bic = bic[-1]
		gmm = gmm_tmp
		print('lowest_bic: '+str(i)+' components')
		print(lowest_bic)
### plot gmm BIC
# x value
xpos=np.array(range(1,len(bic)+1))
# y value
bic = np.array(bic)
# plt
plt.scatter(xpos, bic)
plt.xlim( (0, bic.shape[0]+2) )
plt.ylim( (np.min(bic)-10000, np.max(bic)+10000) )
plt.savefig('gmm_bic.pdf')
plt.close()
###
print('GMM Done!')
### plot GMM model
histdist = plt.hist(data0_sample, 1000, normed=True, histtype='bar',color='k')
plotgauss1 = lambda x: plt.plot(x,(gmm.weights_[0])*mlb.normpdf(x,gmm.means_[0],np.sqrt(gmm.covariances_[0]))[0], linewidth=1.0,color='0.75',linestyle='--')
plotgauss1(histdist[1])
y_predict_fun = lambda x: (gmm.weights_[0])*mlb.normpdf(x,gmm.means_[0],np.sqrt(gmm.covariances_[0]))[0]
y_predict = y_predict_fun(histdist[1])
for i in range(1,len(gmm.means_)):
	plotgauss1 = lambda x: plt.plot(x,(gmm.weights_[i])*mlb.normpdf(x,gmm.means_[i],np.sqrt(gmm.covariances_[i]))[0], linewidth=1.0,color='0.75',linestyle='--')
	plotgauss1(histdist[1])
	y_predict_fun = lambda x: (gmm.weights_[i])*mlb.normpdf(x,gmm.means_[i],np.sqrt(gmm.covariances_[i]))[0]
	y_predict = y_predict + y_predict_fun(histdist[1])
plt.plot(histdist[1], y_predict, 'b--', linewidth=1.0)
plt.savefig('gmm.fit_curve.pdf')
plt.close()
################################################################################################
### get GMM labels
data0_flat = data0.flatten()
data0_flat = data0_flat.reshape(data0_flat.shape[0],1)
labels = gmm.predict(data0_flat)
print('labels.shape')
print(labels.shape)
print(labels[0:10])
################################################################################################
### split rnn hidden state based on GMM cluster
gmm_cm_num = np.unique(labels)
print('gmm_cm_num')
print(gmm_cm_num)

labels_kmeans_label = np.array(labels, dtype=str)

for i in gmm_cm_num:
	print('gmm model kmeans: '+str(i))
	rnn_h_npz0_cm = rnn_h_npz0[labels==i,:]
	print('rnn_h_npz0_cm.shape')
	print(rnn_h_npz0_cm.shape)
	### kmeans clustering
	kmeans_k=3
	kmeans = KMeans(n_clusters=kmeans_k, init='k-means++', max_iter=100, n_init=1, verbose=0, random_state=2017).fit(rnn_h_npz0_cm)
	kmeans_label=kmeans.labels_
	print('kmeans_label.shape:')
	print(kmeans_label.shape)

	used_id = np.where(labels_kmeans_label==str(i))
	for j,k in zip(used_id[0], range(rnn_h_npz0_cm.shape[0])):
		#print(labels_kmeans_label[j])
		labels_kmeans_label[j] = labels_kmeans_label[j]+'_'+str(kmeans_label[k])

print(labels_kmeans_label[0:10])
labels_kmeans_label_matrix = (labels_kmeans_label.reshape(labels_kmeans_label.shape[0]/100,100))
write2d_array(labels_kmeans_label_matrix,'labels_kmeans_label_matrix.txt')
################################################################################################










