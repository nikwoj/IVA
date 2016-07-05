import matplotlib.pyplot as plt

from scipy.io import loadmat
from numpy import corrcoef, zeros, arange, abs
from sys import argv

def plot_corr_comp_numpy(comp, subjects) :
    B = make_data(comp, subjects)
    
    fig, ax = plt.subplots()
    heatmap = plt.pcolor(B, cmap=plt.cm.hot)
    ax.set_xticks(arange(B.shape[0])+0.5, minor=False)
    ax.set_yticks(arange(B.shape[1])+0.5, minor=False)
    
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    
    cbar = plt.colorbar(heatmap)
    plt.show()
    plt.savefig("numpy_heatmap_comp%d.png"%comp)


def make_data(comp, subjects) :
    _,R = loadmat(subjects[0])["S"].shape
    K = len(subjects)
    X = zeros((K, R))
    
    for k in range(K) :
        X[k,:] = loadmat(subjects[k])["S"][comp,:]
    
    return abs(corrcoef(X))

if __name__ == "__main__" :
    argv.pop(0)
    comp = int(argv.pop(0)) - 1
    plot_corr_comp_numpy(comp, argv)
