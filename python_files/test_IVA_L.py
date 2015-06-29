import IVA_L_class as iva
import IVA_L as IVA

import scipy.io as sio
import numpy as np

if __name__ == "__main__" :
    a = sio.loadmat("variables.mat")
    S = a['S']
    X = S.copy()
    N,T,K = S.shape
    A = np.random.rand(N,N,K)
    
    for k in range(K) :
        A[:,:,k] = iva._vecnorm(A[:,:,k])
        X[:,:,k] = np.dot(A[:,:,k],S[:,:,k])
        
    np.random.seed(0)
    W = IVA.IVA_L(X)
    
    np.random.seed(0)
    test = iva.IVA_L()
    test.fit(X)
    W_c = test.unmixing_()
    
    ## If same, should just return matrix of zeros for every site
    for k in range(K) :
        print np.dot(W[:,:,k], A[:,:,k]) - np.dot(W_c[:,:,k], A[:,:,k]), "\n"