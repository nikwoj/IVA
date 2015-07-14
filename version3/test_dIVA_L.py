import dIVA_L as diva
import scipy.io as sio
import numpy as np
import joint_isi as isi
import IVA_L as iva

if __name__ == "__main__" :
    a  = sio.loadmat("d_variables.mat")
    Sm = a['S']
    
    N,T,K = Sm.shape
    
    ## Now comparing with distributed_version
    Am = np.random.rand(N,N,K)
    X = np.zeros(shape=(N,T,K))
    for k in range(K) :
        X[:,:,k] = np.dot(Am[:,:,k], Sm[:,:,k])
    
    W, _, _ = iva.iva_l(X, W_init=[])
    print "Joint ISI for normal IVA_L \t", isi.joint_ISI(W,Am)
    
    
    ## Now comparing with distributed version
    P = 4
    K = 5
    X = np.zeros(shape=(N,T,K,P))
    A = np.zeros(shape=(N,N,K,P))
    S = np.zeros(shape=(N,T,K,P))
    
    for k in range(5) :
        for p in range(4) :
            S[:,:,k,p] = Sm[:,:,k+5*p]
            A[:,:,k,p] = Am[:,:,k+5*p]
    
    for p in range(P) :
        for k in range(K) :
            X[:,:,k,p] = np.dot(A[:,:,k,p], S[:,:,k,p])
    
    W, _, _ = diva.diva_l(X, W_init=[])
    print "Joint ISI for distributed IVA \t", isi.joint_ISI(W,A)