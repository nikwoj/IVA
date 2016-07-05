import numpy as np
import matplotlib.pyplot as plt

from numpy import sum, zeros, sqrt, abs, arange
from scipy.io import loadmat

from sys import argv

from import_SCV_data import get_S_data

'''
 NOTE : To use, call with python in following manner: 

   python make_corr_heatmap.py component_number data_to_use

Data should be organized as a number of different mat files
(one for each different subject) and be able to be called
as loadmat(data1)['S']
'''

def plot_corr_comp(comp, num_subj) :
    B = get_sum_squared_mat(comp, num_subj)
    B = abs(get_corr_comp_mat(B))
    fig, ax = plt.subplots()
    heatmap = plt.pcolor(B, cmap=plt.cm.hot)
    ax.set_xticks(arange(0, B.shape[0], int(B.shape[0] / 3.0)), minor=False)
    ax.set_yticks(arange(0, B.shape[1], int(B.shape[1] / 3.0)), minor=False)
    
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.axis('tight')
    
    cbar = plt.colorbar(heatmap)
    
    plt.show()
    
    plt.savefig("heatmap_comp%d.png"%comp)

def get_corr_comp_mat(B) :
    K, K = B.shape
    R = B.copy()
    for k in range(K) :
        R[k,k] = 1
        
        for kk in range(k+1,K) :
            R[k,kk] = B[k,kk] / sqrt(B[k,k] * B[kk,kk])
            R[kk,k] = R[k,kk]
    
    return R


def get_sum_squared_mat(comp, num_subj) :
    K = num_subj
    B = zeros((K,K))
    
    for k in range(K) :
        #X1 = loadmat(subjects[k])['S'][comp,:]
        X1 = get_S_data([k+1])[comp,:]
        X1 -= X1.mean()
        B[k,k] = sum((X1)**2)
        
        for kk in range(k, K) :
            #X2 = loadmat(subjects[kk])['S'][comp-1,:]
            X2 = get_S_data([kk+1])[comp,:]
            X2 -= X2.mean()
            B[k,kk] = sum(X1 * X2)
            
            B[kk,k] = B[k,kk]
    
    return B * (1.0/(K-1))


if __name__ == "__main__" :
    argv.pop(0)
    comp = int(argv.pop(0))
    num_subj = int(argv.pop(0))
    plot_corr_comp(comp, num_subj)
