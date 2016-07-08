#!/bin/bash

echo "Test 1 has two sites with increasing number of subjects per site"

SUBJS=$1
SEED=$2

echo "$SUBJS $SEED" > seed_subjs.txt

python set_seed_subj_rand_proj.py $SUBJS $SEED

# Assumes pre-processing has been done (in form of PCA dim reduction)
matlab -nodesktop -nodisplay -nosplash -r IVAG_pooled_rand_proj
echo "Finished IVAG"

echo "Running pooled IVAL"
python test_IVAG_pooled_rand_proj.py $SUBJS $SEED 

echo "Running pooled IVAL with no IVAG"
python test_noIVAG_pooled_rand_proj.py $SUBJS $SEED

# This is what form the data should be in
# SCV_IVA_caseNik_r001_pcawhitened_subj[0-9][0-9][0-9][0-9].mat 
#               A_IVA_caseNik_r001_subj[0-9][0-9][0-9][0-9].mat

