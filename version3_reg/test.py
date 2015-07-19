import numpy as np
import joint_isi as isi
import IVA_L as iva
import scipy.io as sio


def main(scale) :
    S = sio.loadmat("d_variables.mat")['S']
    N,T,K = S.shape
    K=2
    A = np.random.rand(N,N,K)
    
    X = np.zeros((N,T,K))
    for k in range(K) :
        X[:,:,k] = np.dot(A[:,:,k], S[:,:,k])
    
    W_initial = A.copy()
    for k in range(K) :
        W_initial[:,:,k] = np.linalg.pinv(A[:,:,k]) + scale * np.random.rand(N,N)
    
    test = iva.IVA_L(verbose=True, W_init=W_initial)
    test.fit(X)
    W = test.W
    
    for k in range(K) :
        print np.dot(W[:,:,k], A[:,:,k])
    
    print "Joint ISI is %f" % isi.joint_ISI(A,W)

if __name__ == "__main__" :
    # main(float(raw_input("Scale factor?")))
    main(.5)