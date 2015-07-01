import numpy    as np
import scipy.io as sio

import dIVA_L as diva

if __name__ == "__main__" :
    a = sio.loadmat("d_variables.mat")
    S = a['S']
    
    N,T,K,P = S.shape
    
    mixing_site = np.random.rand(N,N,K,P)
    
    X = np.zeros(shape=(N,T,K,P))
    
    for p in range(P) :
        for k in range(K) :
            X[:,:,k,p] = np.dot(mixing_site[:,:,k,p], S[:,:,k,p])
    
    unmixing_site = diva.diva_l(X, verbose=True, A=mixing_site)
    
    for p in range(P) :
        print '\n\n'
        for k in range(K) :
            print np.dot(unmixing_site[:,:,k,p], mixing_site[:,:,k,p])