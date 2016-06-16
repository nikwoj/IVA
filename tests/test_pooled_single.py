from multiprocessing import Pool

from numpy import dot, zeros, ceil, abs, arange
from numpy.random import rand
from scipy.io import loadmat
from IVA_L import iva_l
from joint_isi import joint_isi

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

def save_isi(number, write, cost) :
    fil = open("test_ieee_1_pooled_%i.txt"%number, "w")
    fil.write(write + "\n")
    for i in range(2048) :
        fil.write(str(cost[i]) + "\n")
    fil.close()

def algorithm(num_subj) :
    X, A = get_data(num_subj)
    ncomp = 20
    W = rand(ncomp,ncomp,num_subj)
    W, Wht, de_wht, cost = iva_l(X, W, n_components=ncomp, verbose=True)
    cost = de_wht_cost[1]
    isi = joint_isi(W,A,Wht)
    save_isi(num_subj, str((num_subj,isi)), cost)
    print (num_subj, isi)
    return num_subj, isi

def get_data(up) :
    X = zeros((250, 32968,up))
    A = zeros((250,20,up))
    for k in range(1, up+1) :
        A[:,:,k-1] = loadmat("A_IVA_caseNik_r001_subj" + rem_index(k) + ".mat")['A']
        X[:,:,k-1] = dot(A[:,:,k-1], loadmat("SCV_IVA_caseNik_r001_subj" + rem_index(k)+".mat")['S'])
    return X, A

if __name__=="__main__" :
    ncomp = 20
    num_subj = int(argv[1])
    algorithm(num_subj)
