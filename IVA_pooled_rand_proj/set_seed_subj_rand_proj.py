from numpy import dot, zeros
from numpy.random import permutation, seed, rand

from sys import argv
from scipy.io import savemat

from import_SCV_data import get_X_data
## Writes a file with the names of the subjects to use. Matlab will
## then read this file and thus, Matlab and python are using same data


def main (subjs, set_seed) :
    seed(0)
    proj = zeros((20,250))
    permute = permutation(250)[:20]
    for n in range(20) :
        proj[n,permute[n]] = 1
    
    list_subjs = range(1, subjs+1)
    X = get_X_data(list_subjs)
    Y = zeros((20,32968,subjs))
    for k in range(subjs) :
        Y[:,:,k] = dot(proj, X[:,:,k])
    
    seed(set_seed)
    W = rand(20,20,subjs)
    savemat("SCV_IVA_rand_proj_W_subj%d_start%d.mat"%(subjs, set_seed), {'X':Y, 'W':W, 'proj':proj})

def index4 (k) :
    if k < 10 : return "000" + str(k)
    if k < 100 : return "00" + str(k)
    if k < 1000 : return "0" + str(k)
    else : return str(k)

if __name__ == "__main__" :
    subjs = int(argv.pop(1))
    set_seed = int(argv.pop(1))
    main(subjs, set_seed)
