from numpy.random import permutation, seed
from sys import argv
from scipy.io import savemat

from import_SCV_data import get_X_white_data
## Writes a file with the names of the subjects to use. Matlab will
## then read this file and thus, Matlab and python are using same data


def main (set_seed, subjs) :
    seed(set_seed)
    permute = permutation(1024)[0:subjs] + 1
    X = get_X_white_data(permute)
    savemat("SCV_IVA_pcawhitened_seed%d_subj%d.mat"%(set_seed, subjs), {'X_white':X})

def index4 (k) :
    if k < 10 : return "000" + str(k)
    if k < 100 : return "00" + str(k)
    if k < 1000 : return "0" + str(k)
    else : return str(k)

if __name__ == "__main__" :
    set_seed = int(argv.pop(1))
    subjs = int(argv.pop(1))
    main(set_seed, subjs)
