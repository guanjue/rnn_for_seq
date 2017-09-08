import os,sys
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
def read2d_array(filename,dtype_used):
	data=open(filename,'r')
	data0=[]
	for records in data:
		tmp = [x.strip() for x in records.split('\t')]
		data0.append(tmp)
	data0 = np.array(data0,dtype=dtype_used)
	data.close()
	return data0
def write2d_array(array,output):
	r1=open(output,'w')
	for records in array:
		for i in range(0,len(records)-1):
			r1.write(str(records[i])+'\t')
		r1.write(str(records[len(records)-1])+'\n')
	r1.close()
################################################################################################
### read np array
labels_gmm_kmeans = read2d_array('labels_kmeans_label_matrix.txt',str)
sequence = read2d_array('x_matrix_digit.txt',int)
h_pred = read2d_array('rnn_hidden_layer_pred_softmax.txt',float)

print('labels_gmm_kmeans.shape')
print(labels_gmm_kmeans.shape)
print('sequence.shape')
print(sequence.shape)
print('h_pred.shape')
print(h_pred.shape)
################################################################################################
network = {}
network_superstate = {}
pred_cluster = {}
pred_supercluster = {}
t_s = {}
t_s_super = {}
edge_num = {}
for i in range(labels_gmm_kmeans.shape[0]):
	for j in range(2, labels_gmm_kmeans.shape[1]-2):
		s_node = labels_gmm_kmeans[i,j]
		t_node = labels_gmm_kmeans[i,j+1]
		#edge = sequence[i,j+1]
		#edge = [sequence[i,j], sequence[i,j+1], sequence[i,j+2]]
		#edge = [sequence[i,j], sequence[i,j+1]]
		#edge = [sequence[i,j-2], sequence[i,j-1], sequence[i,j], sequence[i,j+1]]
		edge = [sequence[i,j-2], sequence[i,j-1], sequence[i,j], sequence[i,j+1], sequence[i,j+2]]
		s_pred = h_pred[i,j]
		t_pred = h_pred[i,j+1]
		### append network
		if s_node+str(edge)+t_node in network:
			network[s_node+str(edge)+t_node][3] = network[s_node+str(edge)+t_node][3]+1
		else:
			network[s_node+str(edge)+t_node] = [s_node, edge, t_node, 1]
		### append network superstate
		if s_node.split('_')[0]+str(edge)+t_node.split('_')[0] in network_superstate:
			network_superstate[s_node.split('_')[0]+str(edge)+t_node.split('_')[0]][3] = network_superstate[s_node.split('_')[0]+str(edge)+t_node.split('_')[0]][3]+1
		else:
			network_superstate[s_node.split('_')[0]+str(edge)+t_node.split('_')[0]] = [s_node.split('_')[0], edge, t_node.split('_')[0], 1]
		### append pred dict (s_node)
		if s_node in pred_cluster:
			pred_cluster[s_node].append(s_pred)
		else:
			pred_cluster[s_node] = [s_pred]
		### append pred dict (t_node)
		if t_node in pred_cluster:
			pred_cluster[t_node].append(t_pred)
		else:
			pred_cluster[t_node] = [t_pred]
		### append super state pred dict (s_node)
		if s_node.split('_')[0] in pred_supercluster:
			pred_supercluster[s_node.split('_')[0]].append(s_pred)
		else:
			pred_supercluster[s_node.split('_')[0]] = [s_pred]
		### append super state pred dict (t_node)
		if t_node.split('_')[0] in pred_supercluster:
			pred_supercluster[t_node.split('_')[0]].append(t_pred)
		else:
			pred_supercluster[t_node.split('_')[0]] = [t_pred]
		### append nodes dict (s_t)
		if s_node+t_node in t_s:
			t_s[s_node+t_node] = t_s[s_node+t_node] + 1
		else:
			t_s[s_node+t_node] = 1
		### append nodes dict (s_t) superstate
		if s_node.split('_')[0]+t_node.split('_')[0] in t_s_super:
			t_s_super[s_node.split('_')[0]+t_node.split('_')[0]] = t_s_super[s_node.split('_')[0]+t_node.split('_')[0]] + 1
		else:
			t_s_super[s_node.split('_')[0]+t_node.split('_')[0]] = 1
		### append edge dict 
		if str(edge) in edge_num:
			edge_num[str(edge)] = edge_num[str(edge)] + 1
		else:
			edge_num[str(edge)] = 1		
################################################################################################
seq = ['A','C','G','T']
dimer_list = {}
h=0
for i in range(0,4):
	for j in range(0,4):
		for k in range(0,4):
			dimer_list[str(h)] = seq[i]+seq[j]+seq[k]
			h=h+1
### convrt dict to array
### for network
network_matrix = []
for records in network:
	if network[records][3] >=0:
		#print(network[records][1])
		kmer_tmp = seq[(network[records][1][0])]
		for d in range(1,len(network[records][1])):
			kmer_tmp = kmer_tmp + seq[(network[records][1][d])]
		### get expected edge number by chance
		exp_set_num = float(t_s[network[records][0]+network[records][2]] * edge_num[str(network[records][1])]) / float(labels_gmm_kmeans.shape[0] * labels_gmm_kmeans.shape[1])
		network_matrix.append([network[records][0], kmer_tmp, network[records][2], network[records][3], exp_set_num, (network[records][3]+100)/(exp_set_num+100) ])
network_matrix = np.array(network_matrix)
network_matrix = network_matrix[np.argsort(np.array(network_matrix[:,3],dtype=int))]
write2d_array(network_matrix,'network_table.txt')
################################################################################################
### prepare to plot network_enrichment_hist histgram
print('network_matrix[:,5].shape')
print(network_matrix[:,5].shape)
network_enrichment_forhist = np.array(network_matrix[:,5],dtype=float)
print(network_enrichment_forhist[0:10])
################################################################################################
### get significant network edge
network_matrix_thresh = []
z_score = stats.norm.ppf(0.99)
for records in network_matrix:
	if float(records[5])>=np.mean(network_enrichment_forhist)+z_score*np.std(network_enrichment_forhist):
		network_matrix_thresh.append(records)
network_matrix_thresh = np.array(network_matrix_thresh)
network_matrix_thresh = network_matrix_thresh[np.argsort(np.array(network_matrix_thresh[:,3],dtype=int))]
write2d_array(network_matrix_thresh,'network_matrix_thresh.txt')
################################################################################################
### plot network_enrichment_hist histgram
plt.hist(network_enrichment_forhist, 100, normed=True, histtype='bar',color='k')
plt.axvline(np.mean(network_enrichment_forhist)+z_score*np.std(network_enrichment_forhist), color='b', linestyle='dashed', linewidth=2)
plt.savefig('network_enrichment_hist_state.pdf')
################################################################################################
### get significant network edge
network_matrix_thresh = []
z_score = stats.norm.ppf(0.99)
for records in network_matrix:
	if float(records[5])>=(np.mean(network_enrichment_forhist)+z_score*np.std(network_enrichment_forhist)):
		if records[0] != records[2]:
			network_matrix_thresh.append(records)
network_matrix_thresh = np.array(network_matrix_thresh)
network_matrix_thresh = network_matrix_thresh[np.argsort(np.array(network_matrix_thresh[:,3],dtype=int))]
write2d_array(network_matrix_thresh,'network_matrix_thresh_noself.txt')
################################################################################################
### for prediction
pred_cluster_table = []
for records in pred_cluster:
	pred_all = np.array(pred_cluster[records])
	pred_cluster_table.append( [records, np.mean(pred_all)] )
write2d_array(pred_cluster_table,'pred_cluster_table.txt')
################################################################################################
network_superstate_matrix = []
for records in network_superstate:
	if network_superstate[records][3] >=0:
		#print(network[records][1])
		### dimer digit to dimer sequence
		kmer_tmp = seq[(network_superstate[records][1][0])]
		for d in range(1,len(network_superstate[records][1])):
			kmer_tmp = kmer_tmp + seq[(network_superstate[records][1][d])]
		### get expected edge number by chance
		exp_set_num = float(t_s_super[network_superstate[records][0]+network_superstate[records][2]] * edge_num[str(network_superstate[records][1])] ) / float(labels_gmm_kmeans.shape[0] * labels_gmm_kmeans.shape[1] )
		network_superstate_matrix.append([network_superstate[records][0], kmer_tmp, network_superstate[records][2], network_superstate[records][3], exp_set_num, (network_superstate[records][3]+100)/(exp_set_num+100) ])
network_superstate_matrix = np.array(network_superstate_matrix)
network_superstate_matrix = network_superstate_matrix[np.argsort(np.array(network_superstate_matrix[:,3],dtype=int))]
write2d_array(network_superstate_matrix,'network_superstate_table.txt')
################################################################################################
### prepare to plot enrichment histgram
print('network_superstate_matrix[:,5].shape')
print(network_superstate_matrix[:,5].shape)
network_enrichment_forhist_superstate = np.array(network_superstate_matrix[:,5],dtype=float)
print(network_enrichment_forhist_superstate[0:10])
################################################################################################
### get significant enriched threshold 0.99
print('network_enrichment_forhist_superstate analysis:')
z_score = stats.norm.ppf(0.99)
print('z score: '+str(z_score))
print('mean: '+str(np.mean(network_enrichment_forhist_superstate)))
print('std: '+str(np.std(network_enrichment_forhist_superstate)))
print('upper lim: '+str(np.mean(network_enrichment_forhist_superstate)+z_score*np.std(network_enrichment_forhist_superstate)))
print('lower lim: '+str(np.mean(network_enrichment_forhist_superstate)-z_score*np.std(network_enrichment_forhist_superstate)))
### only significant network
network_superstate_matrix_thresh = []
for records in network_superstate_matrix:
	if float(records[5])>=np.mean(network_enrichment_forhist_superstate)+z_score*np.std(network_enrichment_forhist_superstate):
		network_superstate_matrix_thresh.append(records)
write2d_array(network_superstate_matrix_thresh,'network_superstate_matrix_thresh.txt')
network_superstate_matrix_thresh_noself = []
################################################################################################
### plot enrichment histgram
plt.hist(network_enrichment_forhist_superstate, 100, normed=True, histtype='bar',color='k')
plt.savefig('network_enrichment_hist_superstate.pdf')
################################################################################################
### noself network
for records in network_superstate_matrix:
	if (float(records[5])>=np.mean(network_enrichment_forhist_superstate)+z_score*np.std(network_enrichment_forhist_superstate)): #or (float(records[5])<=np.mean(enrichment)-z_score*np.std(enrichment)):
		if records[0] != records[2]:
			network_superstate_matrix_thresh_noself.append(records)
write2d_array(network_superstate_matrix_thresh_noself,'network_superstate_matrix_thresh_noself.txt')
### for super state prediction
pred_supercluster_table = []
for records in pred_supercluster:
	pred_all = np.array(pred_supercluster[records])
	pred_supercluster_table.append( [records, np.mean(pred_all)] )
write2d_array(pred_supercluster_table,'pred_supercluster_table.txt')
################################################################################################

