from IVA_L import iva_l
from joint_isi import joint_isi
from import_SCV_data import get_X_white_data, get_A_data, get_wht_data

from numpy import zeros, dot, isnan
from numpy.random import permutation, seed
from scipy.io import loadmat, savemat
from sys import argv

def main(num_subj, set_seed) :
    seed(set_seed)
    # + 1 to get indices from 1 to 1024
    shift = permutation(1024)[:num_subj] + 1
    
    #X, W = get_data(num_subj, shift)
    X = get_X_white_data(shift)
    W = loadmat("W_IVA_G_si1_su%d_site1.mat"%num_subj)['W']
    
    W, cost = iva_l(X,W,verbose=True)
    
    #A, wht = get_A_wht(num_subj, shift)
    A = get_A_data(shift)
    wht = get_wht_data(shift)
    
    isi = joint_isi(W,A,wht)
    print "(isi, num_subj) : %f %d"%(isi, num_subj)
    save_stuff(num_subj, W, isi, cost, shift, set_seed)

#def get_data(num_subj, shift) :
#    X = zeros((20,32968,num_subj))
#    W = loadmat("W_IVA_G_si1_su%d_site1.mat"%num_subj)['W']
#    for k in range(num_subj) :
#        X[:,:,k] = loadmat("SCV_IVA_caseNik_r001_pcawhitened_subj%s.mat"%index(shift[k]))['X_white']
#    
#    return X, W

#def get_A_wht(num_subj, shift) :
#    A   = zeros((250,20,num_subj))
#    wht = zeros((20,250,num_subj))
#    for k in range(num_subj) :
#        A[:,:,k] = loadmat("A_IVA_caseNik_r001_subj%s.mat"%index(shift[k]))['A']
#        wht[:,:,k] = loadmat("SCV_IVA_caseNik_r001_pcawhitened_subj%s.mat"%index(shift[k]))['wht']
#    
#    return A, wht

def save_stuff(num_subj, W, isi, cost, shift, set_seed) :
    fil = open("test_IVAG_pooled_shift%s_subj%s.txt"%(index(set_seed), index(num_subj)), 'w')
    fil.write(str(isi) + ", " + str(num_subj) + "\n")
    fil.write("Shift\n")
    for i in shift :
        fil.write(str(i) + "\n")
    fil.write("Cost\n")
    for i in cost :
        fil.write(str(i) + "\n")
    fil.close()

def index(num) :
    if num < 10 :
        return "000" + str(num)
    elif num < 100 :
        return "00" + str(num)
    elif num < 1000 :
        return "0" + str(num)
    else :
        return str(num)

if __name__ == "__main__" :
    num_subj = int(argv.pop(1))
    set_seed = int(argv.pop(1))
    main(num_subj, set_seed)
