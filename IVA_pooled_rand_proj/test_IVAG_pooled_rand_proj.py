#from IVAL_back import iva_l
from IVAL_backtrack import iva_l
from joint_isi import joint_isi

import numpy as np

from numpy import zeros, dot, isnan
from scipy.io import loadmat, savemat
from sys import argv, float_info
from import_SCV_data import get_A_data

def main(num_subj, set_seed) :
    X, W, proj = get_data(num_subj, set_seed)
    X, W = normalize_X_W(X,W)
    
    list_subj = range(1, num_subj+1)
    A = get_A_data(list_subj)
    print joint_isi(W,A,proj)
    
    #np.random.seed(0)
    #W = np.random.rand(20,20,num_subj)
    W, cost = iva_l(X,W,verbose=True,term_threshold=1e-8)
    
    isi = joint_isi(W,A,proj)
    print "(isi, num_subj, start) : %f %d %d"%(isi, num_subj, set_seed)
    save_stuff(num_subj, set_seed, W, proj, isi, cost)

def normalize_X_W(X,W) :
    N,_,K = X.shape
    for k in xrange(K) :
        for n in xrange(N) :
            X[n,:,k] -= X[n,:,k].mean()
    
    #Y = X.copy()
    #for k in range(K) :
    #    Y[:,:,k] = dot(W[:,:,k], X[:,:,k])
    #    for n in range(N) :
    #         W[n,:,k] = 1 / (np.std(Y[n,:,k]) + float_info.epsilon) * np.sqrt(K-1)
    
    return X, W
    

def get_data(num_subj, set_seed) :
    data = loadmat("SCV_IVA_rand_proj_W_subj%d_start%d.mat"%(num_subj, set_seed))
    X    = data['X']
    proj = data['proj']
    
    W = loadmat("W_IVAG_pooled_rand_proj_subj%d_start%d.mat"%(num_subj, set_seed))['W']
    return X, W, proj

def save_stuff(num_subj, set_seed, W, proj, isi, cost) :
    fil = open("test_IVAG_pooled_subj%s_start%s.txt"%(index4(num_subj), index2(set_seed)), 'w')
    fil.write(str(isi) + ", " + str(num_subj) + ", " + str(set_seed) + "\n")
    for i in cost :
        fil.write(str(i) + "\n")
    fil.close()
    
    savemat("test_IVAG_pooled_subj%s_start%s.mat"%(index4(num_subj), index2(set_seed)), {'W':W, 'proj':proj})

def index4(num) :
    if num < 10 : return "000" + str(num)
    if num < 100 : return "00" + str(num)
    if num < 1000 : return "0" + str(num)
    else : return str(num)

def index2(num) : 
    if num < 10 : return "0" + str(num)
    else : return str(num)

if __name__ == "__main__" :
    num_subj = int(argv.pop(1))
    set_seed = int(argv.pop(1))
    main(num_subj, set_seed)
