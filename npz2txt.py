import numpy as np

def write2d_array(array,output):
	r1=open(output,'w')
	for records in array:
		for i in range(0,len(records)-1):
			r1.write(str(records[i])+'\t')
		r1.write(str(records[len(records)-1])+'\n')
	r1.close()


data_test0 = np.load('test.npz')
data_test = []
for i in data_test0:
	data_test.append(data_test0[i])
xs_test = data_test[1]
ys_test = np.array([data_test[2]])
print(ys_test)
write2d_array(xs_test,'test.txt')
write2d_array(ys_test,'test_y.txt')

data_train0 = np.load('train.npz')
data_train = []
for i in data_test0:
	data_train.append(data_train0[i])
xs_train = data_train[1]
ys_train = np.array([data_train[2]])

write2d_array(xs_train,'train.txt')
write2d_array(ys_train,'train_y.txt')
