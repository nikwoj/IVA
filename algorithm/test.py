from multiprocessing import Pool

from numpy import dot, zeros, ceil
from scipy.io import loadmat
from local_node import local_node


from sys import argv

def algorithm(X_A_ncomp) :
    W, Wht, de_wht = ddiva(X_A_ncomp[0])
    K = X_A_ncomp[1][0].shape[2]
    print (K, joint_isi(W,A,wht))

def get_data() :
    X = zeros((250, 32968,1024))
    A = zeros((250,20,1024))
    for i in range(1024) :
        A[:,:,i] = loadmat(filename_A)
        X[:,:,i] = dot(A[:,:,i], loadmat(filename_s))
    return X, A

def mk_AX_data(A, X, up) :
    X_data = [local_node(X[:,:,0:up]), local_node(X[:,:,up:2*up])]
    A_data = [A[:,:,0:up], A[:,:,up:2*up]]
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
    
    X, A = get_data()
    # for k in range(512/PNUM) :
    #     X[:,:,PNUM*k:PNUM*(k+1)], A[:,:,PNUM*k:PNUM*(k+1)] = get_data(k, PNUM)
    for it in range(512/PNUM) :
        pool.fmap(algorithm, [(mk_AX_data(X, A, k), ncomp) for k in range(it-PNUM, it, 2)])
        pool.join()