from multiprocessing import Pool

from numpy import dot, zeros, ceil
from numpy.random import seed, rand
from scipy.io import loadmat
from local_node import local_node
from ddiva import ddiva
from joint_isi import joint_disi

from sys import argv

def rem_index(k) :
    if k < 10 :
        return "000%d" % k
    if k < 100 :
        return "00%d" % k
    if k < 1000 :
        return "0%d" % k
    else : 
        return "%d" % k

def algorithm(num_subj) :
    X, A = get_data(num_subj)
    X, A = mk_AX_data(X, A, num_subj)
    ncomp = 20
    
    seed(0)
    W = [rand(ncomp,ncomp,2) for k in range(num_subj/2)]
    W, Wht, de_wht = ddiva(X, W, ncomp)
    
    disi = joint_disi(W,A,Wht)
    save_isi(num_subj, disi)
    return num_subj, disi

def save_isi(number, write) :
    fil = open("test_2_ieee_%i.txt"%number, "w+")
    fil.write(write)
    fil.close()

def get_data(up) :
    X = zeros((250, 32968,up))
    A = zeros((250,20,up))
    for k in range(1, up) :
        A[:,:,k-1] = loadmat("A_IVA_caseNik_r001_subj" + rem_index(k) + ".mat")['A']
        X[:,:,k-1] = dot(A[:,:,k-1], loadmat("SCV_IVA_caseNik_r001_subj" + rem_index(k) + ".mat")['S'])
    return X, A

def mk_AX_data(X, A, up) :
    X_data = [local_node(X[:,:,k:k+2]) for k in range(0,up,2)]
    A_data = [A[:,:,k:k+2], for k in range(0,up,2)]
    return X_data, A_data

if __name__=="__main__" :
    if argv[1] == "liebnitz" :
        PNUM = 32
    elif argv[1] == "hooke" :
        PNUM = 32
    elif argv[1] == "mars" :
        PNUM = 32
    ncomp = 20
    pool = Pool(PNUM)
    
    B = range(4, 130, 2)
    isi = pool.map(algorithm, B)
    fil = open("test_2_ieee_all.txt", "w+")
    for val in isi :
        fil.write(val + "\n")
