from multiprocessing import Pool

from numpy import dot, zeros, ceil, abs
from numpy.random import rand, seed, choice
from scipy.io import loadmat
from local_node import local_node
from ddiva import ddiva
from joint_isi import joint_disi

from sys import argv
from time import sleep

import numpy as np

## Randomly chooses X random subjects from a sample of 1024
## and aplies the ddiva algorithm to them. X is user supplied

def rem_index(k) :
    if k < 10 :
        return "000%d" % k
    if k < 100 :
        return "00%d" % k
    if k < 1000 :
        return "0%d" % k
    else : 
        return "%d" % k

def save_isi(number, index, write) :
    fil = open("test_ieee_single_%i.txt"%number, "w")
    fil.write(write + "\n" + index)
    fil.close()

def algorithm(num_subj) :
    X, A, index = get_data(num_subj)
    X, A = mk_AX_data(X, A, num_subj/2)
    ncomp = 20
    W = [rand(ncomp,ncomp,num_subj/2), rand(ncomp,ncomp,num_subj/2)]
    W, Wht, de_wht = ddiva(X, W, ncomp, verbose=True)
    K = num_subj/2
    isi = joint_disi(W,A,Wht)
    save_isi(num_subj/2, str(index), str((K,isi)))
    print (K, isi)
    print index
    return str((K, isi)), str(index)

def get_data(up) :
    index = choice(np.arange(1, 1025), up, replace=False)
    X = zeros((250,32968,up))
    A = zeros((250,20,up))
    j = 0
    for k in index :
        A[:,:,j-1] = loadmat("A_IVA_caseNik_r001_subj" + rem_index(k) + ".mat")['A']
        X[:,:,j-1] = dot(A[:,:,j-1], loadmat("SCV_IVA_caseNik_r001_subj"+(rem_index(k)+".mat"))['S'])
        j += 1
    return X, A, index

def mk_AX_data(X, A, up) :
    X_data = [local_node(X[:,:,0:up]), local_node(X[:,:,up:])]
    A_data = [A[:,:,0:up], A[:,:,up:]]
    return X_data, A_data

if __name__=="__main__" :
    num = int(argv[1])
    algorithm(num)
