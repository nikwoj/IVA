from IVA_L import iva_l
from joint_isi import joint_isi

from numpy import zeros, dot, isnan
from scipy.io import loadmat, savemat
from sys import argv

def main(num_subj) :
    X, W = get_data(num_subj)
    
    W, cost = iva_l(X,W,verbose=True)
    
    A, wht = get_A_wht(num_subj)
    
    isi = joint_isi(W,A,wht)
    print "(isi, num_subj) : %f %d"%(isi, num_subj)
    save_stuff(num_subj, W, isi, cost)

def get_data(num_subj) :
    X = zeros((20,32968,num_subj))
    W = loadmat("W_IVA_G_si1_su%d_site1.mat"%num_subj)['W']
    for k in range(num_subj) :
        X[:,:,k] = loadmat("SCV_IVA_caseNik_r001_pcawhitened_subj%s.mat"%index(k+1))['X_white']
    
    return X, W

def get_A_wht(num_subj) :
    A   = zeros((250,20,num_subj))
    wht = zeros((20,250,num_subj))
    for k in range(num_subj) :
        A[:,:,k] = loadmat("A_IVA_caseNik_r001_subj%s.mat"%index(k+1))['A']
        wht[:,:,k] = loadmat("SCV_IVA_caseNik_r001_pcawhitened_subj%s.mat"%index(k+1))['wht']
    
    return A, wht

def save_stuff(num_subj, W, isi, cost) :
    fil = open("test_IVAG_pooled_subj%s.txt"%index(num_subj), 'w')
    fil.write(str(isi) + ", " + str(num_subj) + "\n")
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
    main(num_subj)
