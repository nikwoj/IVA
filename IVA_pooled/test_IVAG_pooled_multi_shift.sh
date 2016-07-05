#!/bin/bash

function test_func { 
    cd ~/IVA_ENV/bin
    source activate
    cd ../IVA/testing
    subjs=$1
    seed=$2
    
    echo "$subjs $seed" > seed_subjs.txt
    
    python set_seed_subj.py $seed $subjs
    
    # Assumes pre-processing has been done (in form of PCA dim reduction)
    matlab -nodesktop -nodisplay -nosplash -r IVAG_pooled_shift
    echo "Finished IVAG"
    
    echo "Running pooled IVAL"
    python test_pooled_IVAG_shift.py $subjs $seed
    
    echo "Running pooled IVAL with no IVAG"
    python test_pooled_noIVAG_shift.py $subjs $seed
    
    # This is what form the data should be in
    # SCV_IVA_caseNik_r001_pcawhitened_subj[0-9][0-9][0-9][0-9].mat 
    #               A_IVA_caseNik_r001_subj[0-9][0-9][0-9][0-9].mat
}

for j in $(seq 1 20);
do
    echo "a"
    test="test"$j
    screen -dmS $test test_func 4 $j
done

#echo "Test 1 has two sites with increasing number of subjects per site"
#read -p "Number of subjects to use = " subjs
#read -p "Seed to start with = " seed
