import numpy as np

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
### dict to pred matrix
def dict2matrix(data_dict):
	import numpy as np
	data_matrix = []
	for records in data_dict:
		sample = np.array(data_dict[records])
		pred_cluster_table.append( [records, np.mean(pred_all)] )
	return pred_cluster_table
##########################################################
def NW(s1,s2,match = 1,mismatch = -1, gap = -2):
	penalty = {'MATCH': match, 'MISMATCH': mismatch, 'GAP': gap} #A dictionary for all the penalty valuse.
	n = len(s1) + 1 #The dimension of the matrix columns.
	m = len(s2) + 1 #The dimension of the matrix rows.
	al_mat = np.zeros((m,n),dtype = int) #Initializes the alighment matrix with zeros.
	p_mat = np.zeros((m,n),dtype = str) #Initializes the alighment matrix with zeros.
	#Scans all the first rows element in the matrix and fill it with "gap penalty"
	for i in range(m):
		al_mat[i][0] = penalty['GAP'] * i
		p_mat[i][0] = 'V'
	#Scans all the first columns element in the matrix and fill it with "gap penalty"
	for j in range (n):
		al_mat[0][j] = penalty['GAP'] * j
		p_mat [0][j] = 'H'
	#Fill the matrix with the correct values.
	p_mat [0][0] = 0 #Return the first element of the pointer matrix back to 0.
	for i in range(1,m):
		for j in range(1,n):
			di = al_mat[i-1][j-1] + Diagonal(s1[j-1],s2[i-1],penalty) #The value for match/mismatch -  diagonal.
			ho = al_mat[i][j-1] + penalty['GAP'] #The value for gap - horizontal.(from the left cell)
			ve = al_mat[i-1][j] + penalty['GAP'] #The value for gap - vertical.(from the upper cell)
			al_mat[i][j] = max(di,ho,ve) #Fill the matrix with the maximal value.(based on the python default maximum)
			p_mat[i][j] = Pointers(di,ho,ve)
	#print np.matrix(al_mat)
	#print np.matrix(p_mat)
	return al_mat, p_mat
##########################################################
def Diagonal(n1,n2,pt):
	if(n1 == n2):
		return pt['MATCH']
	else:
		return pt['MISMATCH']
##########################################################
def Pointers(di,ho,ve):
	pointer = max(di,ho,ve) #based on python default maximum(return the first element).
	if(di == pointer):
		return 'D'
	elif(ho == pointer):
		return 'H'
	else:
		 return 'V' 
##########################################################
def pairwise_align(seq1, seq2):
	### NW alignment socrematrix & pointer matrix
	score_m, p_m = NW(seq1,seq2,match = 2,mismatch = -1, gap = -1)
	align_score = np.max(score_m)
	### initialize aligned sequence
	seq1_a = ''
	seq2_a = ''
	### get first pointer position by max score 
	pos_now = [p_m.shape[0]-1, p_m.shape[1]-1]
	### get first pointer
	p_now = p_m[pos_now[0], pos_now[1]]
	pos1 = 0
	pos2 = 0
	while p_now !='0': ### check if reach the source position
		if p_now == 'D': ### if coming from Diagonal
			pos1 = pos1 - 1
			pos2 = pos2 - 1
			seq1_a = seq1_a + seq1[pos1]
			seq2_a = seq2_a + seq2[pos2]	   
			pos_now = [pos_now[0]-1, pos_now[1]-1] ### update next position
		elif p_now == 'H': ### if coming from horizontal.(from the left cell) 
			pos1 = pos1 - 1
			seq1_a = seq1_a + seq1[pos1]
			seq2_a = seq2_a + '-'  
			pos_now = [pos_now[0], pos_now[1]-1] ### update next position
		elif p_now == 'V': ### if coming from vertical.(from the upper cell)
			pos2 = pos2 - 1
			seq1_a = seq1_a + '-'
			seq2_a = seq2_a + seq2[pos2]
			pos_now = [pos_now[0]-1, pos_now[1]] ### update next position
		### update pointer
		p_now = p_m[pos_now[0], pos_now[1]]
	### reverse aligned sequence
	seq1_a = seq1_a[::-1]
	seq2_a = seq2_a[::-1]
	### get merged sequence
	seq12_merge = ''
	for i, j in zip(seq1_a, seq2_a):
		if i == j: ### match
			seq12_merge = seq12_merge + i
		elif i=='-': ### gap
			seq12_merge = seq12_merge + j
		elif j=='-': ### gap
			seq12_merge = seq12_merge + i
		else: ### mismatch
			seq12_merge = seq12_merge + '-'
	return seq1_a, seq2_a, seq12_merge, align_score
##########################################################
def multi_seq2alignedseq(seqlist):
	seqlist0 = seqlist ### cp seqlist for realign
	k=0
	while seqlist.shape[0] != 0:
		#print('step: '+str(k))
		if k>=20:
			print('something is not right...')
			break
		print(seqlist)
		best_score = -100 ### step k best align score
		if k==0:
			for i in range(len(seqlist)-1): ### pairwise compare
				for j in range(i+1,len(seqlist)):
					seq1 = seqlist[i]
					seq2 = seqlist[j]
					seq1_a, seq2_a, seq12_merge, align_score = pairwise_align(seq1, seq2) ### pairwise NW align
					if align_score > best_score:
						best_score = align_score
						best_merge_seq = seq12_merge
						best_ids = [i, j]
		if k!=0:
			for i in range(len(seqlist)):
				seq1 = seqlist[i] ### sequence 1
				seq2 = best_merge_seq ### sequence 2 use the best merged sequence
				seq1_a, seq2_a, seq12_merge, align_score = pairwise_align(seq1, seq2)
				if align_score > best_score:
					best_score = align_score
					best_merge_seq = seq12_merge
					best_ids = [i]
		k = k+1
		### remove best aligned 2 sequence from original sequence list
		seqlist_new = []
		for i in range(seqlist.shape[0]):
			if not (i in best_ids):
				seqlist_new.append(seqlist[i])
		seqlist = np.array(seqlist_new)

		#print('best align pair: '+str(best_ids))
		#print('best align score: '+str(best_score))
		#print('best merged align seq: '+best_merge_seq)

	### realign all sequence 2 best merged seq
	print('best_merge_seq: '+best_merge_seq)
	seqlist0_allaligned = []
	for records in seqlist0:
		seq1 = records
		seq2 = best_merge_seq
		seq1_a, seq2_a, seq12_merge, align_score = pairwise_align(seq1, seq2)
		seqlist0_allaligned.append(seq1_a)
	print('all aligned sequence: ')
	for records in seqlist0_allaligned:
		print(records)
	return seqlist0_allaligned, best_merge_seq
##########################################################
def align_motif(inputname, outputname):
	superstate_matrix = read2d_array(inputname,str)
	superstate_matrix_edge = {}
	superstate_matrix_info = {}

	for records in superstate_matrix:
		name = records[0]+'-'+records[2]
		### convert matrix to edge dict
		if name in superstate_matrix_edge:
			superstate_matrix_edge[name].append(records[1])
		else:
			superstate_matrix_edge[name] = [records[1]]
		### convert matrix to info dict
		if name in superstate_matrix_info:
			superstate_matrix_info[name].append(records)
		else:
			superstate_matrix_info[name] = [records]

	result_matrix = []
	result_matrix_best_aligned_edge = []
	for records in superstate_matrix_edge:
		if np.array(superstate_matrix_edge[records]).shape[0]>1:
			print('state_pair: '+records)
			aligned_edges, best_all_algned_edge = multi_seq2alignedseq(np.array(superstate_matrix_edge[records]))
		else: ### if only 1 edge, no align
			aligned_edges, best_all_algned_edge = superstate_matrix_edge[records], superstate_matrix_edge[records][0]
		### get new matrix with aligned edge
		for info, al_edge in zip(superstate_matrix_info[records], aligned_edges):
			result_matrix.append([info[0], al_edge, info[2], info[3], info[4], info[5]])
		### extract best aligned edge matrix
		info_matrix = superstate_matrix_info[records]
		obs = np.sum(np.array(np.array(info_matrix)[:,3],dtype=float))
		exp = np.sum(np.array(np.array(info_matrix)[:,4],dtype=float))
		fc = obs / exp
		result_matrix_best_aligned_edge.append([records.split('-')[0], best_all_algned_edge, records.split('-')[1], obs, exp, fc])
	write2d_array(result_matrix, outputname)
	write2d_array(result_matrix_best_aligned_edge, outputname+'.best.txt')
##########################################################


align_motif('network_superstate_matrix_thresh_noself.txt', 'network_superstate_matrix_thresh_noself_aligned.txt')
align_motif('network_matrix_thresh_noself.txt', 'network_matrix_thresh_noself_aligned.txt')

k=5
for i in range(k):
	align_motif('network_table_pos_kmeans/network_superstate_matrix_thresh_noself_pos_'+str(i)+'.txt', 'network_table_pos_kmeans/network_superstate_matrix_thresh_noself_pos_'+str(i)+'_aligned.txt')
	align_motif('network_table_pos_kmeans/network_matrix_thresh_noself_pos_'+str(i)+'.txt', 'network_table_pos_kmeans/network_matrix_thresh_noself_pos_'+str(i)+'_aligned.txt')




