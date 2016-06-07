import numpy as np
import matplotlib.pyplot as plt

from numpy import sum, zeros, sqrt
from scipy.io import loadmat

from sys import argv, float_info


def plot_corr_comp(comp, subjects) :
    B = get_sum_squared_mat(comp, subjects)
    B = get_corr_comp_mat(B)
    
    fig, ax = plt.subplots()
    heatmap = plt.pcolor(B, cmpa=plt.cm.gray)
    ax.set_xticks(np.arange(B.shape[0])+0.5, minor=False)
    ax.set_yticks(np.arange(B.shape[1])+0.5, minor=False)
    
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    
    cbar = plt.colorbar(heatmap)
    plt.show()
    fig.savefig("heatmap_comp%d.png"%comp)

def get_corr_comp_mat(B) :
    K, K = B.shape
    R = B.copy()
    for k in range(K) :
        R[k,k] = 1
        
        for kk in range(k+1,K) :
            R[k,kk] = B[k,kk] / (sqrt(B[k,k] * B[kk,kk]) + float_info.epsilon)
            R[kk,k] = R[k,kk]
    
    return R


def get_sum_squared_mat(comp, subjects) :
    K = len(subjects)
    B = zeros((K,K))
    for k in range(K) :
        X1 = loadmat(subjects[k])['S'][comp-1,:]
        X1 -= X1.mean()
        B[k,k] = sum((X1)**2)
        
        for kk in range(k+1, K) :
            X2 = loadmat(subjects[kk])['S'][comp-1,:]
            X2 -= X2.mean()
            B[k,kk] = sum(X1*X2)
            B[kk,k] = B[k,kk]
    
    return B * (1/K-1)


if __name__ == "__main__" :
    argv.pop(0)
    comp = int(argv.pop(0))
    plot_corr_comp(comp, argv)
