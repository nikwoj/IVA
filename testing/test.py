from master_node import master_node
from local_node import local_node
from joint_isi import joint_ISI
from scipy.io import loadmat

import numpy as np

def main() :
    N = 3
    R = 10000
    P = 5
    K = 4
    
    Sm = loadmat("d_variables")['S']
    S = [np.zeros((N,R,K)) for p in range(P)]
    A = []
    X = np.zeros((N,R,K))
    list_of_sites = []
    for p in range(P) :
        A.append( np.random.randn(3,3,4) )
        for k in range(K) :
            S[p][:,:,k] = Sm[:,:,p*K + k]
        list_of_sites.append( local_node() )
        
        for k in range(K) :
            X[:,:,k] = np.dot(A[p][:,:,k], S[p][:,:,k])
        
        list_of_sites[p]._fit(X)
    
    master = master_node(verbose=True)
    W = master.fit (list_of_sites)
    
    print W[0].shape
    
    for p in range(P) :
        print joint_ISI(W[p], A[p])
    

if __name__ == "__main__" :
    main()
    