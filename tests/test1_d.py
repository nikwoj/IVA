from multiprocessing import Pool

from numpy import dot, zeros, abs, arange, identity
from numpy.random import rand, seed
from scipy.io import loadmat, savemat
from local_node import local_node
from ddiva import ddiva
from joint_isi import joint_disi

from sys import argv
from time import sleep

# : NOTE : To use, call with 3 numbers. num_threads num_start num_stop
## num_servers : How many threads to have going at one time
## num_start : How many subjects should we start with?
## num_stop : How many subjects should we end with?
## ## If num_servers < 2 then will do a single test, and num_start will be number
## ## of subjects  to use.
## num_start should be less than num_stop. Both should be even
## All numbers should be integers
##
## Example call : python test1_d.py 20 4 180
## 
## Use 20 different threads, start with 4 subjects, end with 180 subjects
##
## Assumes data is in form 
##
##  A_IVA_caseNik_r001_subj(index).mat ['A']
##  SCV_IVA_caseNik_r001_subj(index).mat ['S']


def rem_index(k) :
    if k < 10 :
        return "000%d" % k
    if k < 100 :
        return "00%d" % k
    if k < 1000 :
        return "0%d" % k
    else : 
        return "%d" % k

def algorithm1(num_subj, Verbose=False) :
    X, A = get_data1(num_subj)
    X, A = mk_AX_data1(X, A, num_subj/2)
    ncomp = 20
    
    seed(0)
    W = get_W1(num_subj/2.0, ncomp)
    W, Wht, de_wht, cost = ddiva(X, W, ncomp, verbose=Verbose)
    
    
    K = num_subj
    isi = joint_disi(W,A,Wht)
    save_stuff1(str(K), str((K,isi)), W, Wht, de_wht, cost)
    print (K, isi)
    return K, isi

def save_stuff1(num, write, W, Wht, de_wht, cost) :
    fil = open("test_1_ieee_" + num + ".txt", "w")
    fil.write(write + "\n")
    
    for i in range(len(cost)) :
        fil.write(str(cost[i]) + "\n")
    fil.close()
    
    savemat("test_1_ieee_" + num + "_matrices.mat", {"W":W,"Wht":Wht,"de_wht":de_wht})

def get_data1(up) :
    X = zeros((250, 32968,up))
    A = zeros((250,20,up))
    for k in range(1, up+1) :
        A[:,:,k-1] = loadmat("A_IVA_caseNik_r001_subj" + rem_index(k) + ".mat")['A']
        X[:,:,k-1] = dot(A[:,:,k-1], loadmat("SCV_IVA_caseNik_r001_subj" + rem_index(k) + ".mat")['S'])
    return X, A

def mk_AX_data1(X, A, up) :
    X_data = [local_node(X[:,:,0:up]), local_node(X[:,:,up:])]
    A_data = [A[:,:,0:up], A[:,:,up:]]
    return X_data, A_data

def get_W1(subjs, ncomp) :
     W = [zeros((ncomp, ncomp, subjs)), zeros((ncomp, ncomp, subjs))]
     
     for k in range(2) :
         for kk in range(int(subjs)) :
             W[k][:,:,kk] = identity(ncomp) #rand(ncomp, ncomp)
     
     return W

if __name__=="__main__" :
    if int(argv[1]) < 2 :
        subjects = int(argv[2])
        algorithm1(subjects, Verbose=True)
    else :
        PNUM = int(argv[1])
    
        pool = Pool(PNUM)
        
        start = int(argv[2])
        stop  = int(argv[3])
        
        #B = arange(start,stop,2)
        B = [2**i for i in range(2, 11)]
        isi = pool.map(algorithm1, B)
        fil = open("test_1_ieee_all.txt", "w+")
        fil.write(isi)
        fil.close()
