from numpy import array, arange, zeros, sum, sqrt
import matplotlib.pyplot as plt
from make_corr_heatmap import get_corr_comp_mat

from numpy.random import rand

def test_heatmap() :
    data = array([[1,4,7],[6,5,4],[9,5,1]])
    
    fig, ax = plt.subplots()
    heatmap = plt.pcolor(data, cmap=plt.cm.hot)
    ax.set_xticks(arange(data.shape[0]) + 0.5, minor=False)
    ax.set_yticks(arange(data.shape[1]) + 0.5, minor=False)
    
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    
    cbar = plt.colorbar(heatmap)
    plt.show()
    
    plt.savefig("test_heatmap.png")

def get_sum_squared_mat(data) :
    K = data.shape[0]
    B = zeros((K,K))
    
    for k in range(K) :
        X1 = data[k,:]
        X1 -= X1.mean()
        B[k,k] = sum(X1**2)
        
        for kk in range(k+1, K) :
            X2 = data[kk,:]
            X2 -= X2.mean()
            B[k,kk] = sum(X1*X2)
            B[kk,k] = B[k,kk]
    
    return (1/(K-1)) * B




if __name__=="__main__" :
    test_heatmap()
