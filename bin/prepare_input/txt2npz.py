import numpy as np

################################################################################################
def read2d_array(filename):
	data0=open(filename,'r')
	data = []
	for records in data0:
		tmp = [x.strip() for x in records.split('\t')]
		data.append(tmp)
	data0.close()
	return data


def txt2npz(file_pos_train, file_pos_test, file_neg_train, file_neg_test, output_train, output_test):
	pos_train = np.array(read2d_array(file_pos_train),dtype=int)
	pos_test = np.array(read2d_array(file_pos_test),dtype=int)
	neg_train = np.array(read2d_array(file_neg_train),dtype=int)
	neg_test = np.array(read2d_array(file_neg_test),dtype=int)
	### concatenate pos & neg
	train_x_data = np.concatenate((pos_train, neg_train), axis=0)
	test_x_data = np.concatenate((pos_test, neg_test), axis=0)
	train_y_data = np.concatenate((np.repeat(1,pos_train.shape[0]), np.repeat(0,neg_train.shape[0])), axis=0)
	test_y_data = np.concatenate((np.repeat(1,pos_test.shape[0]), np.repeat(0,neg_test.shape[0])), axis=0)
	### save npz
	np.savez(output_train, name=output_train, data=train_x_data, label=train_y_data)
	np.savez(output_test, name=output_test, data=test_x_data, label=test_y_data)

############################################################################
#time python txt2npz.py -a train_pos.txt -b test_pos.txt -c train_neg.txt -d test_neg.txt -e train.npz -f test.npz

import getopt
import sys
def main(argv):
	try:
		opts, args = getopt.getopt(argv,"ha:b:c:d:e:f:")
	except getopt.GetoptError:
		print 'python txt2npz.py -a file_pos_train -b file_pos_test -c file_neg_train -d file_neg_test -e output_train -f output_test'
		sys.exit(2)

	for opt,arg in opts:
		if opt=="-h":
			print 'python txt2npz.py -a file_pos_train -b file_pos_test -c file_neg_train -d file_neg_test -e output_train -f output_test'
			sys.exit()
		elif opt=="-a":
			file_pos_train=str(arg.strip())
		elif opt=="-b":
			file_pos_test=str(arg.strip())
		elif opt=="-c":
			file_neg_train=str(arg.strip())
		elif opt=="-d":
			file_neg_test=str(arg.strip())
		elif opt=="-e":
			output_train=str(arg.strip())
		elif opt=="-f":
			output_test=str(arg.strip())

	txt2npz(file_pos_train, file_pos_test, file_neg_train, file_neg_test, output_train, output_test)

if __name__=="__main__":
	main(sys.argv[1:])
