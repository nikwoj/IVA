import dIVA_L as diva
import scipy.io as sio
import numpy as np
import joint_isi as isi

if __name__ == "__main__" :
    a  = sio.loadmat("d_variables.mat")
    Sm = a['S']
    
    N,T,K = Sm.shape
    
    S = np.zeros(shape=(N,T,5,4))
    
    for k in range(5) :
        for p in range(4) :
            S[:,:,k,p] = Sm[:,:,k+5*p]
        
    N,T,K,P = S.shape
    X = np.zeros(shape=(N,T,K,P))
    A = np.random.rand(N,N,K,P)
    
    for p in range(P) :
        for k in range(K) :
            X[:,:,k,p] = np.dot(A[:,:,k,p], S[:,:,k,p])
    
    W, _, _ = diva.diva_l(X, verbose=True)
    
    for p in range(P) :
        for k in range(K) :
            print np.dot(W[:,:,k,p], A[:,:,k,p]), "\n"
    
    print isi.joint_ISI(W,A)