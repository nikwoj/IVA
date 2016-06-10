from multiprocessing import Pool

from numpy import dot, zeros, ceil, abs
from numpy.random import rand
from scipy.io import loadmat
from local_node import local_node
from ddiva import ddiva
from joint_isi import joint_disi

from sys import argv
from time import sleep

def rem_index(k) :
    if k < 10 :
        return "000%d" % k
    if k < 100 :
        return "00%d" % k
    if k < 1000 :
        return "0%d" % k
    else : 
        return "%d" % k

def save_isi(number, write) :
    fil = open("test_ieee_single_%i.txt"%number, "w")
    fil.write(write)
    fil.close()

def algorithm(num_subj) :
    X, A = get_data(num_subj)
    X, A = mk_AX_data(X, A, num_subj/2)
    ncomp = 20
    W = [rand(ncomp,ncomp,num_subj/2), rand(ncomp,ncomp,num_subj/2)]
    W, Wht, de_wht = ddiva(X, W, ncomp, verbose=True)
    K = num_subj/2
    isi = joint_disi(W,A,Wht)
    save_isi(num_subj/2, str((K,isi)))
    print (K, isi)
    return str((K, isi))

def get_data(up) :
    X = zeros((250, 32968,up))
    A = zeros((250,20,up))
    for k in range(1, up+1) :
        A[:,:,k-1] = loadmat("A_IVA_caseNik_r001_subj" + rem_index(k) + ".mat")['A']
        X[:,:,k-1] = dot(A[:,:,k-1], loadmat("SCV_IVA_caseNik_r001_subj"+(rem_index(k)+".mat"))['S'])
    return X, A

def mk_AX_data(X, A, up) :
    X_data = [local_node(X[:,:,0:up]), local_node(X[:,:,up:])]
    A_data = [A[:,:,0:up], A[:,:,up:]]
    return X_data, A_data

if __name__=="__main__" :
    num = int(argv[1])
    algorithm(num)
