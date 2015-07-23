from optimize import iva_l
from joint_isi import joint_ISI

from scipy.io import loadmat
from autograd.numpy.random import rand
from autograd.numpy.linalg import pinv

import numpy as np

def main( scale ) :
    S = loadmat("d_variables.mat")['S'][:,0:200,:]
    N,R,K = S.shape
    K = 2
    X = np.zeros((N,R,K))
    A = rand(N,N,K)
    
    for k in range(K) :
        X[:,:,k] = np.dot(A[:,:,k], S[:,:,k])
    
    try :
        scale = float(scale)
        W = np.zeros((N,N,K))
        for k in range(K) :
            W[:,:,k] = pinv(A[:,:,k]) + scale * rand(N,N)
            print W.shape
    except :
        W = rand(N,N,K)
    
    W = iva_l(X, W_init=W, verbose=True)
    
    print "Joint ISI is: %f"%joint_ISI(W.x,A)

if __name__ == "__main__" :
    scale = raw_input("Scale factor away from inverse of A? ")
    main( scale )