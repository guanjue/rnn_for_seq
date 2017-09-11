import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import KMeans
import seaborn as sns
################################################################################################
### read 2d array
def read2d_array(filename,dtype_used):
	import numpy as np
	data=open(filename,'r')
	data0=[]
	for records in data:
		tmp = [x.strip() for x in records.split('\t')]
		data0.append(tmp)
	data0 = np.array(data0,dtype=dtype_used)
	data.close()
	return data0
################################################################################################
### write 2d matrix
def write2d_array(array,output):
	r1=open(output,'w')
	for records in array:
		for i in range(0,len(records)-1):
			r1.write(str(records[i])+'\t')
		r1.write(str(records[len(records)-1])+'\n')
	r1.close()
################################################################################################
### convrt dict to array
def netdict_seq2char_netmatrix(netdict, seq_list, edge_num, statepair_dict, labels_gmm_kmeans0, labels_gmm_kmeans):
	netmatrix = []
	for records in netdict:
		### convert seq to character
		kmer_tmp = seq_list[(netdict[records][1][0])]
		for d in range(1,len(netdict[records][1])):
			kmer_tmp = kmer_tmp + seq_list[(netdict[records][1][d])]
		### get expected edge number by chance
		all_edge_num = float(labels_gmm_kmeans0.shape[0] * labels_gmm_kmeans0.shape[1])
		cluster_edge_num = float(labels_gmm_kmeans.shape[0] * labels_gmm_kmeans.shape[1])
		seq_len = labels_gmm_kmeans.shape[1]
		add_small_num = cluster_edge_num/seq_len/50
		exp_set_num = float(statepair_dict[netdict[records][0]+netdict[records][2]] * edge_num[str(netdict[records][1])]) * cluster_edge_num / all_edge_num**2 
		netmatrix.append([netdict[records][0], kmer_tmp, netdict[records][2], netdict[records][3], exp_set_num, (netdict[records][3]+add_small_num)/(exp_set_num+add_small_num) ])
	netmatrix = np.array(netmatrix)	
	print('add_small_num: '+str(add_small_num))
	return netmatrix
################################################################################################
### pred_dict to pred matrix
def pred_dict2pred_matrix(pred_cluster):
	import numpy as np
	pred_cluster_table = []
	for records in pred_cluster:
		pred_all = np.array(pred_cluster[records])
		pred_cluster_table.append( [records, np.mean(pred_all)] )
	return pred_cluster_table
################################################################################################
### get significant network edge
def extract_enriched_edges(netmatrix, net_enrichment_array, significance, enrichement_hist_filename):
	import matplotlib.pyplot as plt
	from scipy import stats
	print('network_enrichment_forhist_superstate analysis:')
	### extract enrichment array
	net_enrichment_forhist = np.log(net_enrichment_array) ### use all sample enrichment to calculate enrichment
	### get significant stat	
	z_score = stats.norm.ppf(significance)
	print('z score: '+str(z_score))
	print('mean: '+str(np.mean(net_enrichment_forhist)))
	print('std: '+str(np.std(net_enrichment_forhist)))
	print('upper lim: '+str(np.mean(net_enrichment_forhist)+z_score*np.std(net_enrichment_forhist)))
	print('lower lim: '+str(np.mean(net_enrichment_forhist)-z_score*np.std(net_enrichment_forhist)))
	print('netmatrix.shape'+str(netmatrix.shape))
	### extract network SET based on threshold
	netmatrix_thresh = []
	netmatrix_thresh_noself = []
	for records in netmatrix:
		if np.log(float(records[5]))>=np.mean(net_enrichment_forhist)+z_score*np.std(net_enrichment_forhist):
			netmatrix_thresh.append(records)
			if records[0] != records[2]:
				netmatrix_thresh_noself.append(records)
	netmatrix_thresh = np.array(netmatrix_thresh)
	netmatrix_thresh_noself = np.array(netmatrix_thresh_noself)
	### plot network enrichment distribution histgram
	plt.hist(net_enrichment_forhist, 100, normed=True, histtype='bar',color='k')
	plt.axvline(np.mean(net_enrichment_forhist)+z_score*np.std(net_enrichment_forhist), color='b', linestyle='dashed', linewidth=2)
	plt.savefig(enrichement_hist_filename)
	plt.close()
	return netmatrix_thresh, netmatrix_thresh_noself

################################################################################################
### read np array
labels_gmm_kmeans0 = read2d_array('labels_kmeans_label_matrix.txt',str)
sequence0 = read2d_array('x_matrix_digit.txt',int)
h_pred0 = read2d_array('rnn_hidden_layer_pos_pred_transpose.txt',float)
print('labels_gmm_kmeans.shape')
print(labels_gmm_kmeans0.shape)
print('sequence.shape')
print(sequence0.shape)
print('h_pred.shape')
print(h_pred0.shape)
################################################################################################
### extract net from input matrixs (label matrix, sequence, hidden_state_before_softmax)
def generate_net(labels_gmm_kmeans, sequence, h_pred):
	net = {}
	net_superstate = {}
	pred_cluster = {}
	pred_supercluster = {}
	t_s = {}
	t_s_super = {}
	edge_num = {}
	for i in range(labels_gmm_kmeans.shape[0]):
		for j in range(2, labels_gmm_kmeans.shape[1]-2):
			s_node = labels_gmm_kmeans[i,j]
			t_node = labels_gmm_kmeans[i,j+1]
			#edge = [sequence[i,j-2], sequence[i,j-1], sequence[i,j], sequence[i,j+1]]
			edge = [sequence[i,j-1], sequence[i,j], sequence[i,j+1], sequence[i,j+2]]
			s_pred = h_pred[i,j]
			t_pred = h_pred[i,j+1]
			### append network
			if s_node+str(edge)+t_node in net:
				net[s_node+str(edge)+t_node][3] = net[s_node+str(edge)+t_node][3]+1
			else:
				net[s_node+str(edge)+t_node] = [s_node, edge, t_node, 1]
			### append network superstate
			if s_node.split('_')[0]+str(edge)+t_node.split('_')[0] in net_superstate:
				net_superstate[s_node.split('_')[0]+str(edge)+t_node.split('_')[0]][3] = net_superstate[s_node.split('_')[0]+str(edge)+t_node.split('_')[0]][3]+1
			else:
				net_superstate[s_node.split('_')[0]+str(edge)+t_node.split('_')[0]] = [s_node.split('_')[0], edge, t_node.split('_')[0], 1]
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
			### append edge dict 
			if str(edge) in edge_num:
				edge_num[str(edge)] = edge_num[str(edge)] + 1
			else:
				edge_num[str(edge)] = 1
	return net, net_superstate, pred_cluster, pred_supercluster, edge_num, t_s, t_s_super

################################################################################################
################################################################################################
################################################################################################
### before sample clustering
network, network_superstate, pred_cluster, pred_supercluster, edge_num, t_s, t_s_super = generate_net(labels_gmm_kmeans0, sequence0, h_pred0)
################################################################################################
### for substate network
#########################
seq = ['A','C','G','T']
### for substate network
network_matrix = netdict_seq2char_netmatrix(network, seq, edge_num, t_s, labels_gmm_kmeans0, labels_gmm_kmeans0)
#network_matrix = network_matrix[np.argsort(np.array(network_matrix[:,3],dtype=int))]
write2d_array(network_matrix,'network_table.txt')
################################################################################################
### get significant substate network edge
network_matrix_thresh, network_matrix_thresh_noself  = extract_enriched_edges(network_matrix, np.array(network_matrix[:,5],dtype=float), 0.99, 'network_enrichment_hist_state.pdf')
### write matrix txt
write2d_array(network_matrix_thresh,'network_matrix_thresh.txt')
write2d_array(network_matrix_thresh_noself,'network_matrix_thresh_noself.txt')
################################################################################################
### for substate prediction
pred_cluster_table = pred_dict2pred_matrix(pred_cluster)
write2d_array(pred_cluster_table,'pred_cluster_table.txt')

################################################################################################
### for superstate network
#########################
seq = ['A','C','G','T']
### for superstate network
network_superstate_matrix = netdict_seq2char_netmatrix(network_superstate, seq, edge_num, t_s_super, labels_gmm_kmeans0, labels_gmm_kmeans0)
#network_matrix = network_matrix[np.argsort(np.array(network_matrix[:,3],dtype=int))]
write2d_array(network_superstate_matrix,'network_superstate_table.txt')
################################################################################################
### get significant superstate network edge
network_superstate_matrix_thresh, network_superstate_matrix_thresh_noself  = extract_enriched_edges(network_superstate_matrix, np.array(network_superstate_matrix[:,5],dtype=float), 0.99, 'network_superstate_enrichment_hist_state.pdf')
### write matrix txt
write2d_array(network_superstate_matrix_thresh,'network_superstate_matrix_thresh.txt')
write2d_array(network_superstate_matrix_thresh_noself,'network_superstate_matrix_thresh_noself.txt')
################################################################################################
### for superstate prediction
pred_supercluster_table = pred_dict2pred_matrix(pred_supercluster)
write2d_array(pred_supercluster_table,'pred_supercluster_table.txt')
################################################################################################

################################################################################################
### only plot pos data
pos_neg = ( (read2d_array('y_obs_matrix.txt',float)[:,0] == 1.0 ) * (read2d_array('y_predict_matrix.txt',float)[:,0] > 0.5 ) ) !=0
labels_gmm_kmeans0_pos = labels_gmm_kmeans0[pos_neg]
sequence0_pos = sequence0[pos_neg]
h_pred0_pos = h_pred0[pos_neg]
### kmean cluster positive sequence
################################################################################################
kmeans_k = 5
print('Kmeans clustering:')
kmeans_pos = KMeans(n_clusters=kmeans_k, init='k-means++', max_iter=100, n_init=1, verbose=0, random_state=2017).fit(h_pred0_pos)
print('Kmeans clustering DONE!')
kmeans_label_pos = kmeans_pos.labels_
plt.figure(figsize=(15, 20))
sns.heatmap(h_pred0_pos[np.argsort(kmeans_label_pos)], center=0, cmap='bwr')
plt.savefig('h_pred_pos_kmeans_heatmap.pdf')
plt.close()

### creat kmeans folder
if not os.path.isdir('network_table_pos_kmeans'):
	os.makedirs('network_table_pos_kmeans')
### kmeans clustering
for k in range(kmeans_k):
	print('kmeans: '+str(k))
	### extract data for kmeans k
	labels_gmm_kmeans0_pos_k = labels_gmm_kmeans0_pos[kmeans_label_pos==k,:]
	sequence0_pos_k = sequence0_pos[kmeans_label_pos==k,:]
	h_pred0_pos_k = h_pred0_pos[kmeans_label_pos==k,:]
	print('h_pred0_pos_k.shape')
	print(h_pred0_pos_k.shape)
	###
	### before sample clustering
	network_pos_k, network_superstate_pos_k, pred_cluster_pos_k, pred_supercluster_pos_k, edge_num_pos_k, t_s_pos_k, t_s_super_pos_k = generate_net(labels_gmm_kmeans0_pos_k, sequence0_pos_k, h_pred0_pos_k)
	#########################
	### for substate network
	network_matrix_pos_k = netdict_seq2char_netmatrix(network_pos_k, seq, edge_num, t_s, labels_gmm_kmeans0, labels_gmm_kmeans0_pos_k)
	print('network_matrix_pos_k.shape: '+str(network_matrix_pos_k.shape))
	#network_matrix = network_matrix[np.argsort(np.array(network_matrix[:,3],dtype=int))]
	write2d_array(network_matrix_pos_k,'network_table_pos_kmeans/network_table_pos_'+str(k)+'.txt')
	################################################################################################
	### get significant substate network edge
	network_matrix_thresh_pos_k, network_matrix_thresh_noself_pos_k  = extract_enriched_edges(network_matrix_pos_k, np.array(network_matrix[:,5],dtype=float), 0.99, 'network_table_pos_kmeans/network_enrichment_hist_state_pos_'+str(k)+'.pdf')
	### write matrix txt
	write2d_array(network_matrix_thresh_pos_k,'network_table_pos_kmeans/network_matrix_thresh_pos_'+str(k)+'.txt')
	write2d_array(network_matrix_thresh_noself_pos_k,'network_table_pos_kmeans/network_matrix_thresh_noself_pos_'+str(k)+'.txt')
	################################################################################################


	################################################################################################
	### for superstate network
	#########################
	### for superstate network
	network_superstate_matrix_pos_k = netdict_seq2char_netmatrix(network_superstate_pos_k, seq, edge_num, t_s_super, labels_gmm_kmeans0, labels_gmm_kmeans0_pos_k)
	print('network_superstate_matrix_pos_k.shape: '+str(network_superstate_matrix_pos_k.shape))
	#network_matrix = network_matrix[np.argsort(np.array(network_matrix[:,3],dtype=int))]
	write2d_array(network_superstate_matrix_pos_k,'network_table_pos_kmeans/network_superstate_table_pos_'+str(k)+'.txt')
	################################################################################################
	### get significant superstate network edge
	network_superstate_matrix_thresh_pos_k, network_superstate_matrix_thresh_noself_pos_k  = extract_enriched_edges(network_superstate_matrix_pos_k, np.array(network_superstate_matrix[:,5],dtype=float), 0.99, 'network_table_pos_kmeans/network_superstate_enrichment_hist_state_pos_'+str(k)+'.pdf')
	### write matrix txt
	write2d_array(network_superstate_matrix_thresh_pos_k,'network_table_pos_kmeans/network_superstate_matrix_thresh_pos_'+str(k)+'.txt')
	write2d_array(network_superstate_matrix_thresh_noself_pos_k,'network_table_pos_kmeans/network_superstate_matrix_thresh_noself_pos_'+str(k)+'.txt')
	################################################################################################


	################################################################################################

