from sys import argv

from scipy.io import loadmat, savemat
from numpy import dot
from 

def algorithm_IVAG(num_subj) :
    X, A, W =  get_data_IVAG(num_subj)
    X, A, W = make_data_IVAG(num_subj)
    
    W, cost = ddiva(X, W, verbose=True)
    Wht = get_Wht(num_subj)
    
    K = num_subj
    isi = joint_disi(W, A, Wht)
    
    save_stuff_IVAG(str(K), str((K, isi)), W, Wht, cost) 
    return K, isi

def save_stuff_IVAG(number, write, W, Wht, cost) :
    
