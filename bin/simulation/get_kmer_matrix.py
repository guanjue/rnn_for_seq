weight_kmerscaner=[]
### get dimer list
dimer_list=[]

kmer_all = []
for i in range(0,4):	
	for p in range(0,5):
		kmer.append([i])
	kmer_all.append(kmer)


print(kmer_all)



for i in range(0,4):
	for j in range(0,4):
		dimer_list.append([i,j])


kmer_k = 3

a = np.zeros((pow(4, kmer_k),kmer_k))

digit = range(4)
for i in range(a.shape[1]):
	k = 0
	m = 0
	for j in range(a.shape[0]):
		if m%(a.shape[0]/pow(4,i+1)) == 0:
			if k ==len(digit):
				k=0
			used_digit = digit[k]
			k = k+1
		a[j,i] = used_digit
		m = m+1



np.array([0]*5)*4

0,1,2,3
1,2,4,

0,1,2
1,2,4

0,1
1,4