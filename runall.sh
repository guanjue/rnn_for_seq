### package location ~/rnn_for_seq/
### train RNN
#time python /Volumes/MAC_Data/data/labs/zhang_lab/sequence_prediction/nfyb_fos/rnn_for_seq/bin/pred_models/rnn.py
### get RNN hidden states for testing data
#time python /Volumes/MAC_Data/data/labs/zhang_lab/sequence_prediction/nfyb_fos/rnn_for_seq/bin/pred_models_info_extract/rnn_analysis.py
### using Gaussian mixture & K means to cluster hidden states
#time python /Volumes/MAC_Data/data/labs/zhang_lab/sequence_prediction/nfyb_fos/rnn_for_seq/bin/gmm_kmeans_network/gmm_kmeans.py
### get network tables
#time python /Volumes/MAC_Data/data/labs/zhang_lab/sequence_prediction/nfyb_fos/rnn_for_seq/bin/gmm_kmeans_network/get_network.py

########################################################
###### simulation analysis
cd /Volumes/MAC_Data/data/labs/zhang_lab/sequence_prediction/nfyb_fos/simulate_data_vs_SHUFFLE
### generate random sequence with 2 motifs motif1 downstream of motif2
time python /Volumes/MAC_Data/data/labs/zhang_lab/sequence_prediction/nfyb_fos/rnn_for_seq/bin/simulation/fasta_generator_coverall.py -l 100 -n 10000 -u ACGT -a 03322  -p 1.0 -b 30331  -q 1.0 -s 2017 -r upstream -o motif1_up_motif2.fa
### dimer shuffle sequence
time fasta-dinucleotide-shuffle -f motif1_up_motif2.fa -s 2017 > motif1_up_motif2.shuffle.fa
### fasta2seq
time python /Volumes/MAC_Data/data/labs/zhang_lab/sequence_prediction/nfyb_fos/rnn_for_seq/bin/prepare_input/fasta2numseq.py -f motif1_up_motif2.fa -o motif1_up_motif2.txt -u ACGT
time python /Volumes/MAC_Data/data/labs/zhang_lab/sequence_prediction/nfyb_fos/rnn_for_seq/bin/prepare_input/fasta2numseq.py -f motif1_up_motif2.shuffle.fa -o motif1_up_motif2.shuffle.txt -u ACGT
### split training data & testing data
head -6000 motif1_up_motif2.txt > train_pos.txt
head -6000 motif1_up_motif2.shuffle.txt > train_neg.txt
tail -n+6001 motif1_up_motif2.txt > test_pos.txt
tail -n+6001 motif1_up_motif2.shuffle.txt > test_neg.txt
### prepare input data (txt2npz)
time python /Volumes/MAC_Data/data/labs/zhang_lab/sequence_prediction/nfyb_fos/rnn_for_seq/bin/prepare_input/txt2npz.py -a train_pos.txt -b test_pos.txt -c train_neg.txt -d test_neg.txt -e train.npz -f test.npz
### train RNN
time python /Volumes/MAC_Data/data/labs/zhang_lab/sequence_prediction/nfyb_fos/rnn_for_seq/bin/pred_models/rnn.py
### get RNN hidden states for testing data
time python /Volumes/MAC_Data/data/labs/zhang_lab/sequence_prediction/nfyb_fos/rnn_for_seq/bin/pred_models_info_extract/rnn_analysis.py
### using Gaussian mixture & K means to cluster hidden states
time python /Volumes/MAC_Data/data/labs/zhang_lab/sequence_prediction/nfyb_fos/rnn_for_seq/bin/gmm_kmeans_network/gmm_kmeans.py
### get network tables
time python /Volumes/MAC_Data/data/labs/zhang_lab/sequence_prediction/nfyb_fos/rnn_for_seq/bin/gmm_kmeans_network/get_network.py

