from scipy.io import loadmat
from os import chdir, getcwd
from numpy import zeros, dot

def get_S_data(list_subj) :
    original = getcwd()
    chdir('/export/mialab/users/rsilva/projects/MultivariateICA/MISA/fresh/data/Sgt/caseNik')
    #chdir('../data')
    K = len(list_subj)
    S = zeros((20,32968,K)) 
    i = 0
    for k in list_subj :
        S[:,:,i] = loadmat("SCV_IVA_caseNik_r001_subj%s.mat"%index4(k))['S']
        i += 1
    
    chdir(original)
    return S

def get_X_data(list_subj) :
    original = getcwd()
    chdir('../data')
    K = len(list_subj)
    X = zeros((250,32968,K))
    i = 0
    A = get_A_data(list_subj)
    S = get_S_data(list_subj)
    for k in list_subj :
        X[:,:,i] = dot(A[:,:,i], S[:,:,i])
        i += 1
    
    chdir(original)
    return X

def get_X_white_data(list_subj) :
    original = getcwd()
    #chdir('/export/mialab/users/rsilva/projects/MultivariateICA/MISA/fresh/data/Sgt/caseNik')
    chdir('../data')
    K = len(list_subj)
    X = zeros((20,32968,K)) 
    i = 0
    for k in list_subj :
        X[:,:,i] = loadmat("SCV_IVA_caseNik_r001_pcawhitened_subj%s.mat"%index4(k))['X_white']
        i += 1
    
    chdir(original)
    return X

def get_A_data(list_subj) :
    original = getcwd()
    #chdir('/export/mialab/users/rsilva/projects/MultivariateICA/MISA/fresh/data/Sgt/caseNik')
    chdir('../data')
    K = len(list_subj)
    A = zeros((250,20,K)) 
    i = 0
    for k in list_subj :
        A[:,:,i] = loadmat("A_IVA_caseNik_r001_subj%s.mat"%index4(k))['A']
        i += 1
    
    chdir(original)
    return A

def get_wht_data(list_subj) :
    original = getcwd()
    #chdir('/export/mialab/users/rsilva/projects/MultivariateICA/MISA/fresh/data/Sgt/caseNik')
    chdir('../data')
    K = len(list_subj)
    wht = zeros((20,250,K)) 
    i = 0
    for k in list_subj :
        wht[:,:,i] = loadmat("SCV_IVA_caseNik_r001_pcawhitened_subj%s.mat"%index4(k))['wht']
        i += 1
    
    chdir(original)
    return wht

def index4 (k) :
    if k < 10 : return "000" + str(k)
    if k < 100 : return "00" + str(k)
    if k < 1000 : return "0" + str(k)
    else : return str(k)
