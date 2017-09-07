def write2d_array(array,output):
	r1=open(output,'w')
	for records in array:
		for i in range(0,len(records)-1):
			r1.write(str(records[i])+'\t')
		r1.write(str(records[len(records)-1])+'\n')
	r1.close()

def fasta2numseq(input_fasta, outputname, nuc0):
	### extract seq unit
	nuc = []
	for s in nuc0:
		nuc.append(s)
	print('sequence chars:')
	print(nuc)
	data0 = open(input_fasta,'r')
	num_seq = []
	for records in data0:
		tmp = [x.strip() for x in records.split('\t')][0]
		if tmp[0] != '>':
			seq = []
			for s in tmp:
				if s == nuc[0]:
					seq.append('0')
				elif s == nuc[1]:
					seq.append('1')			
				elif s == nuc[2]:
					seq.append('2')
				elif s == nuc[3]:
					seq.append('3')
			num_seq.append(seq)	
	data0.close()		
	write2d_array(num_seq, outputname)

############################################################################
#time python fasta2numseq.py -f motif1_up_motif2.fa -o motif1_up_motif2.txt -u ACGT

import getopt
import sys
def main(argv):
	try:
		opts, args = getopt.getopt(argv,"hf:o:u:")
	except getopt.GetoptError:
		print 'python fasta2numseq.py -f fasta_filename -o outputname -u seq_unit<ACGT>'
		sys.exit(2)

	for opt,arg in opts:
		if opt=="-h":
			print 'python fasta2numseq.py -f fasta_filename -o outputname -u seq_unit<ACGT>'
			sys.exit()
		elif opt=="-f":
			input_fasta=str(arg.strip())
		elif opt=="-u":
			nuc0=str(arg.strip())
		elif opt=="-o":
			outputname=str(arg.strip())

	fasta2numseq(input_fasta, outputname, nuc0)
if __name__=="__main__":
	main(sys.argv[1:])
