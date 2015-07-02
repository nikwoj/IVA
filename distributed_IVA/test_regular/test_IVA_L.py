import IVA_L as iva
import scipy.io as sio
import numpy as np
import joint_isi as isi

if __name__ == "__main__" :
    a = sio.loadmat("d_variables.mat")
    sm = a['S']
    N, T, K, P = sm.shape
    S = np.zeros(shape=(N,T,(K*P)))
    
    for p in range(P) :
        for k in range(K) :
            S[:,:,K*p+k] = sm[:,:,k,p]
    
    X = S.copy()
    
    N,T,K = S.shape
    A = np.random.rand(N,N,K)
    
    for k in range(K) :
        X[:,:,k] = np.dot(A[:,:,k],S[:,:,k])
    
    test = iva.IVA_L(verbose=True)
    test.fit(X)
    W = test.unmixing()
    
    for k in range(K) :
        print np.dot(W[:,:,k], A[:,:,k]), "\n"
    
    print isi.joint_ISI(W,A)