### train RNN
time python /Volumes/MAC_Data/data/labs/zhang_lab/sequence_prediction/nfyb_fos/rnn_for_seq/bin/rnn.py
### get RNN hidden states for testing data
time python /Volumes/MAC_Data/data/labs/zhang_lab/sequence_prediction/nfyb_fos/rnn_for_seq/bin/rnn_analysis.py
### using Gaussian mixture & K means to cluster hidden states
time python /Volumes/MAC_Data/data/labs/zhang_lab/sequence_prediction/nfyb_fos/rnn_for_seq/bin/gmm_kmeans.py
### get network tables
time python /Volumes/MAC_Data/data/labs/zhang_lab/sequence_prediction/nfyb_fos/rnn_for_seq/bin/get_network.py
