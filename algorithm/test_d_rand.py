from multiprocessing import Pool

from numpy import dot, zeros, ceil, abs
from numpy.random import rand, choice, seed
from scipy.io import loadmat
from local_node import local_node
from ddiva import ddiva
from joint_isi import joint_disi

from sys import argv
from time import sleep

import numpy as np

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
    fil = open("test_ieee_1_%i.txt"%number, "w")
    fil.write(write + "\n" + index)
    fil.close()

def algorithm(num_subj) :
    X, A, index = get_data(num_subj)
    X, A = mk_AX_data(X, A, num_subj/2)
    ncomp = 20
    W = [rand(ncomp,ncomp,num_subj/2), rand(ncomp,ncomp,num_subj/2)]
    W, Wht, de_wht = ddiva(X, W, ncomp)
    K = num_subj/2
    isi = joint_disi(W,A,Wht)
    save_isi(num_subj/2, str(index), str((K,isi)))
    print (K, isi)
    return K, isi

def get_data(up) :
    index = choice(np.arange(1, 1025), up, replace=False)
    X = zeros((250, 32968,up))
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
    if argv[1] == "liebnitz" :
        PNUM = 20
    elif argv[1] == "hooke" :
        PNUM = 5
    elif argv[1] == "mars" :
        PNUM = 32
    ncomp = 20
    pool = Pool(PNUM)
    
    #X, A = get_data(128)
    #for k in range(512/PNUM) :
    #    X[:,:,PNUM*k:PNUM*(k+1)], A[:,:,PNUM*k:PNUM*(k+1)] = get_data(k, PNUM)
    #for it in range(PNUM+2, 130, 2*PNUM) :
        #pool = Pool(PNUM)
        #B = [k for k in range(it-PNUM,it,2)]
        #print B
        #print pool.map(algorithm, B)
        #pool.close
        #pool.map(algorithm, B)  #[k for k in range(it-PNUM, it, 2)])
        #pool.join()
    B = [k for k in range(4, 130, 2) if k not in [4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,56,60]]
    isi = pool.map(algorithm, B)
    fil = open("test_ieee_all.txt", "w+")
    fil.write(isi)
    fil.close()
