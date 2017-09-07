import numpy as np

def write2d_array(array,output):
	r1=open(output,'w')
	for records in array:
		for i in range(0,len(records)-1):
			r1.write(str(records[i])+'\t')
		r1.write(str(records[len(records)-1])+'\n')
	r1.close()

### seq unit
def fasta_generator(seq_len, seq_num, nuc0, motif1_seq0, motif2_seq0, seed, relative_pos, outputname):
	np.random.seed(seed)
	### extract seq unit
	nuc = []
	for s in nuc0:
		nuc.append(s)
	print('sequence chars:')
	print(nuc)
	### extract target motif
	motif1_seq = []
	for s in motif1_seq0:
		motif1_seq.append(int(s))
	motif2_seq = []
	for s in motif2_seq0:
		motif2_seq.append(int(s))
	print('motif1:')
	print(motif1_seq)
	print('motif2:')
	print(motif2_seq)
	### generate sequence
	print('start simulation!')
	generated_seq = []
	for n in range(seq_num):
		### generate the random sequence
		rand_seq = np.random.randint(4,size=seq_len)

		### randomly select motif 1 position
		if relative_pos == 'upstream':
			motif1_positon = np.random.random_integers(len(motif2_seq), seq_len-1-len(motif1_seq), 1)[0]
		elif relative_pos == 'downstream':
			motif1_positon = np.random.random_integers(0, seq_len-1-len(motif1_seq)-len(motif2_seq), 1)[0]
		else:
			motif1_positon = np.random.random_integers(0, seq_len-1-len(motif1_seq), 1)[0]
		#motif1_positon = np.random.random_integers(0+len(motif2_seq), seq_len-len(motif1_seq)-1, 1)[0]
		### add the primary motif
		i0 = 0
		for i in range(motif1_positon,motif1_positon+len(motif1_seq)):
			rand_seq[i] = motif1_seq[i0]
			i0 = i0+1


		### check if the secondary motif is intersect with the primary motif
		if relative_pos == 'upstream':
			### randomly select motif 2 position
			motif2_positon = np.random.random_integers(0, motif1_positon - len(motif2_seq), 1)[0]
		elif relative_pos == 'downstream':
			### randomly select motif 2 position
			motif2_positon = np.random.random_integers(motif1_positon + len(motif1_seq), seq_len-1-len(motif2_seq), 1)[0]		
		else:
			### randomly select motif 2 position
			motif2_positon = np.random.random_integers(0, seq_len-1-len(motif2_seq), 1)[0]
			while (motif2_positon > motif1_positon - len(motif2_seq)) and (motif2_positon < motif1_positon + len(motif1_seq)):
				motif2_positon = np.random.random_integers(0, seq_len-1-len(motif2_seq), 1)[0]
		### add the second motif
		i0 = 0
		for i in range(motif2_positon,motif2_positon+len(motif2_seq)):
			rand_seq[i] = motif2_seq[i0]
			i0 = i0+1

		### convert acgt to 0123
		generated_seq.append(['>'+str(n)])
		tmp = ''
		for s in rand_seq:
			if s == 0:
				tmp = tmp+nuc[0]
			elif s == 1:
				tmp = tmp+nuc[1]			
			elif s == 2:
				tmp = tmp+nuc[2]
			elif s == 3:
				tmp = tmp+nuc[3]
		generated_seq.append([tmp])
	### write generated fasta
	write2d_array(generated_seq, outputname)

############################################################################
#time python fasta_generator.py -l 100 -n 5000 -u ACGT -a 03322 -b 30331 -s 2017 -r upstream -o motif1_up_motif2.fa

import getopt
import sys
def main(argv):
	try:
		opts, args = getopt.getopt(argv,"hl:n:u:a:b:s:r:o:")
	except getopt.GetoptError:
		print 'python fasta_generator.py -l seq_len -n seq_num -u seq_unit<ACGT> -a motif1<03322> -b motif2<30332> -s random_seed -r relative_position<upstream/downstream/random> -o outputname'
		sys.exit(2)

	for opt,arg in opts:
		if opt=="-h":
			print 'python fasta_generator.py -l seq_len -n seq_num -u seq_unit<ACGT> -a motif1<03322> -b motif2<30332> -s random_seed -r relative_position<upstream/downstream/random> -o outputname'
			sys.exit()
		elif opt=="-l":
			seq_len=int(arg.strip())
		elif opt=="-n":
			seq_num=int(arg.strip())
		elif opt=="-u":
			nuc0=str(arg.strip())
		elif opt=="-a":
			motif1_seq0=str(arg.strip())
		elif opt=="-b":
			motif2_seq0=str(arg.strip())
		elif opt=="-s":
			seed=int(arg.strip())
		elif opt=="-r":
			relative_pos=str(arg.strip())
		elif opt=="-o":
			outputname=str(arg.strip())

	fasta_generator(seq_len, seq_num, nuc0, motif1_seq0, motif2_seq0, seed, relative_pos, outputname)
if __name__=="__main__":
	main(sys.argv[1:])


