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
#cd /Volumes/MAC_Data/data/labs/zhang_lab/sequence_prediction/nfyb_fos/simulate_data_vs_SHUFFLE_3motif
### generate random sequence with 2 motifs motif1 downstream of motif2
#time python /Volumes/MAC_Data/data/labs/zhang_lab/sequence_prediction/nfyb_fos/rnn_for_seq/bin/simulation/fasta_generator_coverall.py -l 100 -n 10000 -u ACGT -a 03322  -p 0.95 -b 30331  -q 0.75 -s 2017 -r upstream -o motif1_up_motif2.fa
### only order
#time python /Volumes/MAC_Data/data/labs/zhang_lab/sequence_prediction/nfyb_fos/rnn_for_seq/bin/simulation/fasta_generator_3motif_coverall.py -l 100 -n 5000 -u ACGT -a 03322 -p 1 -b 10132 -q 1 -c 33301 -t 1 -s 2017 -r upstream -v downstream -o motif1_up_motif2_down_motif3.fa
#time python /Volumes/MAC_Data/data/labs/zhang_lab/sequence_prediction/nfyb_fos/rnn_for_seq/bin/simulation/fasta_generator_3motif_coverall.py -l 100 -n 5000 -u ACGT -a 03322 -p 1 -b 10132 -q 1 -c 33301 -t 1 -s 2017 -r downstream -v upstream -o motif1_up_motif2_down_motif3_noorder.fa
### only order vs random
#time python /Volumes/MAC_Data/data/labs/zhang_lab/sequence_prediction/nfyb_fos/rnn_for_seq/bin/simulation/fasta_generator_3motif_coverall.py -l 100 -n 5000 -u ACGT -a 03322 -p 1 -b 10132 -q 1 -c 33301 -t 1 -s 2017 -r upstream -v downstream -o motif1_up_motif2_down_motif3.fa
#time python /Volumes/MAC_Data/data/labs/zhang_lab/sequence_prediction/nfyb_fos/rnn_for_seq/bin/simulation/fasta_generator_3motif_coverall.py -l 100 -n 5000 -u ACGT -a 03322 -p 1 -b 10132 -q 1 -c 33301 -t 1 -s 2017 -r random -v random -o motif1_up_motif2_down_motif3_noorder.fa

### only enrich 
#time python /Volumes/MAC_Data/data/labs/zhang_lab/sequence_prediction/nfyb_fos/rnn_for_seq/bin/simulation/fasta_generator_3motif_coverall.py -l 100 -n 5000 -u ACGT -a 03322 -p 1 -b 10132 -q 1 -c 33301 -t 1 -s 2017 -r random -v random -o motif1_up_motif2_down_motif3.fa
#time fasta-dinucleotide-shuffle -f motif1_up_motif2_down_motif3.fa -s 2017 > motif1_up_motif2_down_motif3_noorder.fa


### only enrich and order 
#time python /Volumes/MAC_Data/data/labs/zhang_lab/sequence_prediction/nfyb_fos/rnn_for_seq/bin/simulation/fasta_generator_3motif_coverall.py -l 100 -n 5000 -u ACGT -a 03322 -p 1 -b 10132 -q 1 -c 33301 -t 1 -s 2017 -r upstream -v downstream -o motif1_up_motif2_down_motif3.fa
#time fasta-dinucleotide-shuffle -f motif1_up_motif2_down_motif3.fa -s 2017 > motif1_up_motif2_down_motif3_noorder.fa

### dimer shuffle sequence
#time fasta-dinucleotide-shuffle -f motif1_up_motif2_down_motif3.fa -s 2017 > motif1_up_motif2_down_motif3.shuffle.fa

### fasta2seq
#time python /Volumes/MAC_Data/data/labs/zhang_lab/sequence_prediction/nfyb_fos/rnn_for_seq/bin/prepare_input/fasta2numseq.py -f motif1_up_motif2_down_motif3.fa -o motif1_up_motif2_down_motif3.txt -u ACGT
#time python /Volumes/MAC_Data/data/labs/zhang_lab/sequence_prediction/nfyb_fos/rnn_for_seq/bin/prepare_input/fasta2numseq.py -f motif1_up_motif2_down_motif3.shuffle.fa -o motif1_up_motif2_down_motif3.shuffle.txt -u ACGT
#time python /Volumes/MAC_Data/data/labs/zhang_lab/sequence_prediction/nfyb_fos/rnn_for_seq/bin/prepare_input/fasta2numseq.py -f motif1_up_motif2_down_motif3_noorder.fa -o motif1_up_motif2_down_motif3_noorder.txt -u ACGT

### split training data & testing data
#head -3000 motif1_up_motif2_down_motif3.txt > train_pos.txt
head -3000 motif1_up_motif2_down_motif3.fa.posi_list.txt > train_pos.fa.posi_list.txt

#head -6000 motif1_up_motif2_down_motif3.shuffle.txt > train_neg.txt

#head -3000 motif1_up_motif2_down_motif3_noorder.txt > train_neg.txt
head -3000 motif1_up_motif2_down_motif3_noorder.fa.posi_list.txt > train_neg.fa.posi_list.txt

#tail -n+3001 motif1_up_motif2_down_motif3.txt > test_pos.txt
tail -n+3001 motif1_up_motif2_down_motif3.fa.posi_list.txt > test_pos.fa.posi_list.txt

#tail -n+6001 motif1_up_motif2_down_motif3.shuffle.txt > test_neg.txt
#tail -n+3001 motif1_up_motif2_down_motif3_noorder.txt > test_neg.txt
tail -n+3001 motif1_up_motif2_down_motif3_noorder.fa.posi_list.txt > test_neg.fa.posi_list.txt

cat test_pos.txt test_neg.txt > test_all.txt
cat test_pos.fa.posi_list.txt test_neg.fa.posi_list.txt > test_all.fa.posi_list.txt
### prepare input data (txt2npz)
#time python /Volumes/MAC_Data/data/labs/zhang_lab/sequence_prediction/nfyb_fos/rnn_for_seq/bin/prepare_input/txt2npz.py -a train_pos.txt -b test_pos.txt -c train_neg.txt -d test_neg.txt -e train.npz -f test.npz

### train RNN
#time python /Volumes/MAC_Data/data/labs/zhang_lab/sequence_prediction/nfyb_fos/rnn_for_seq/bin/pred_models/rnn.py

### get RNN hidden states for testing data
time python /Volumes/MAC_Data/data/labs/zhang_lab/sequence_prediction/nfyb_fos/rnn_for_seq/bin/pred_models_info_extract/rnn_analysis.py

### using Gaussian mixture & K means to cluster hidden states
time python /Volumes/MAC_Data/data/labs/zhang_lab/sequence_prediction/nfyb_fos/rnn_for_seq/bin/gmm_kmeans_network/gmm_kmeans.py

### get network tables
time python /Volumes/MAC_Data/data/labs/zhang_lab/sequence_prediction/nfyb_fos/rnn_for_seq/bin/gmm_kmeans_network/get_network_clusterseq.py

time python /Volumes/MAC_Data/data/labs/zhang_lab/sequence_prediction/nfyb_fos/rnn_for_seq/bin/gmm_kmeans_network/get_network_clusterseq_neg.py

### NW method align sequence
time python /Volumes/MAC_Data/data/labs/zhang_lab/sequence_prediction/nfyb_fos/rnn_for_seq/bin/gmm_kmeans_network/needleman_wunsch_align.py







paste test_all.fa.posi_list.txt y_obs_matrix.txt y_predict_matrix.txt | awk -F '\t' -v OFS='\t' '{if (($1 > $2) && ($1 < $3)) print 1, $4, $6; else print 0, $4, $6}' > pos_pattern.txt



