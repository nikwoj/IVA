from IVA_L import iva_l
from joint_isi import joint_isi
from import_SCV_data import get_A_data, get_wht_data

from numpy import zeros, dot, identity
from scipy.io import loadmat, savemat
from sys import argv


def main(num_subj, set_seed) :
    X, W = get_data(num_subj, set_seed)
    
    W, cost = iva_l(X,W,verbose=True)
    
    A, wht = get_A_wht(num_subj)
    
    isi = joint_isi(W,A,wht)
    print "(isi, num_subj, start) : %f %d %d"%(isi, num_subj, set_seed)
    save_stuff(num_subj, set_seed, W, isi, cost)

def get_data(num_subj, set_seed) :
    data = loadmat("SCV_IVA_pcawhitened_W_seed%d_subj%d.mat"%(set_seed, num_subj))
    W = data['W']
    X = data['X_white']
    
    return X, W

def get_A_wht(num_subj) :
    list_subj = range(1, num_subj+1)
    A   =   get_A_data(list_subj)
    wht = get_wht_data(list_subj)
    return A, wht

def save_stuff(num_subj, set_seed, W, isi, cost) :
    fil = open("test_noIVAG_pooled_subj%s_start%s.txt"%(index4(num_subj), index2(set_seed)), 'w')
    fil.write(str(isi) + ", " + str(num_subj) + ", " + str(set_seed) + "\n")
    for i in cost :
        fil.write(str(i) + "\n")
    fil.close()
    
    savemat("test_IVAG_pooled_subj%s_start%s.mat"%(index4(num_subj), index2(set_seed)), {'W':W})

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
