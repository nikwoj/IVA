#!/bin/bash

echo "Test 1 has two sites with increasing number of subjects per site"
read -p "Number of sites to use = " num_sites
read -p "Number of subjects per site = " subj_site


echo "$subj_site $num_sites" > num_subj_site.txt

# Assumes pre-processing has been done (in form of PCA dim reduction)
matlab -nodesktop -nodisplay -nosplash -r test1_IVAG_preprocessing
echo "Finished IVAG"

echo "Running ddIVA now"
python test_distributed.py $num_sites $subj_site

echo "Running ddIVA with no IVAG preprocessing now, comparing results"
python test_distributed_noIVAG.py $num_sites $subj_site


# This is what form the data should be in
# SCV_IVA_caseNik_r001_pcawhitened_subj[0-9][0-9][0-9][0-9].mat 
#               A_IVA_caseNik_r001_subj[0-9][0-9][0-9][0-9].mat

