#!/bin/bash

SUBJS=$1
SEED=$2

echo "$SUBJS $SEED" > seed_subjs.txt

ipython set_seed_subj_rand_proj.py $SUBJS $SEED

# Assumes pre-processing has been done (in form of PCA dim reduction)
matlab -nodesktop -nodisplay -nosplash -r IVAG_pooled_rand_proj
echo "Finished IVAG"

echo "Running pooled IVAL"
ipython test_IVAG_pooled_rand_proj.py $SUBJS $SEED 

echo "Running pooled IVAL with no IVAG"
ipython test_noIVAG_pooled_rand_proj.py $SUBJS $SEED

# This is what form the data should be in
# SCV_IVA_caseNik_r001_pcawhitened_subj[0-9][0-9][0-9][0-9].mat 
#               A_IVA_caseNik_r001_subj[0-9][0-9][0-9][0-9].mat

