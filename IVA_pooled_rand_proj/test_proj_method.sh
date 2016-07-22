#!/bin/bash

SUBJS=$1
SEED=$2
METHODS="singular_value qr"
#METHODS="qr"

for METHOD in $METHODS; do
    ipython set_seed_subj_rand_proj.py $SUBJS $SEED $METHOD
    ipython test_proj_method.py $SUBJS $SEED $METHOD
done
