import numpy as np

data=open('test_x_pos.txt','r')
data0=[]
for records in data:
	tmp=records.split('\t')[0:100]
	data0.append(tmp)
data.close()
data0=np.array(data0)
print(data0.shape)

seq_multi_motif = []
for records in data0:
	kmer = '00000'
	#kmer2 = '00000'
	test1=0
	test2=0
	for letter in records:
		kmer = kmer[1:]+letter
		if kmer =='11003' and test2<1:
			test1=test1+1
		if kmer =='03322':#'30331': 33332 13210 32231 33301 20332 01020 03322 11003
			test2=test2+1

	if test1>=1 and test2>=1:
		seq_multi_motif.append(records)

seq_multi_motif = np.array(seq_multi_motif)
print(seq_multi_motif.shape)
