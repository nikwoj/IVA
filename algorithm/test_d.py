from multiprocessing import Pool

from numpy import dot, zeros, ceil, abs
from numpy.random import rand
from scipy.io import loadmat
from local_node import local_node
from ddiva_class_method import ddiva
from joint_isi3 import joint_disi

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
    fil = open("test_ieee_1_%i.txt"%number, "w")
    fil.write(write)
    fil.close()

def algorithm(num_subj) :
    X, A = get_data(num_subj)
    X, A = mk_AX_data(X, A, num_subj/2)
    ncomp = 20
    W = [rand(ncomp,ncomp,num_subj/2), rand(ncomp,ncomp,num_subj/2)]
    W, Wht, de_wht = ddiva(X, W, ncomp)
    K = num_subj/2
    isi = joint_disi(W,A,Wht)
    print (K, isi)
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
    if argv[1] == "liebnitz" :
        PNUM = 20
    elif argv[1] == "hooke" :
        PNUM = 6
    elif argv[1] == "mars" :
        PNUM = 10
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
    #B = [k for k in range(4,8,2)]
    B = [k for k in range(4, 130, 2) if k not in [4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,56,60]]
    isi = pool.map(algorithm, B)
    fil = open("test_ieee_all.txt", "w+")
    for k in range(len(isi)) :
        fil.write(str(isi[k]))
        fil.write("\n")
    fil.close()
