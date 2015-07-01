import IVA_L as iva
import permutation_matrix as per
import scipy.io as sio
import numpy as np
import joint_isi as isi

if __name__ == "__main__" :
    a = sio.loadmat("variables.mat")
    S = a['S']
    X = S.copy()
    
    N,T,K = S.shape
    A = np.random.rand(N,N,K)
    
    for k in range(K) :
        X[:,:,k] = np.dot(A[:,:,k],S[:,:,k])
    
    test = iva.IVA_L(verbose=True)
    test.fit(X)
    W_c = test.unmixing()
    
    for k in range(K) :
        print np.dot(W_c[:,:,k], A[:,:,k]), "\n"
    
    print isi.joint_ISI(W_c,A)