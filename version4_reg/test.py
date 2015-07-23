
import numpy as np

from scipy.io import loadmat
from numpy.linalg import pinv
from numpy.random import rand

from joint_isi import joint_ISI
from IVA_L import iva_l


def main( scale=None, dispersion=[] ) :
    S = loadmat("d_variables.mat")['S']
    X = S.copy()
    N,R,K = S.shape
    
    A = rand(N,N,K)
    W = A.copy()
    
    for k in range(K) :
        X[:,:,k] = np.dot(A[:,:,k], S[:,:,k])
    
    if dispersion=="[]" :
        dispersion = []
    else :
        dispersion = np.zeros((K,K,N))
        for n in range(N) :
            dispersion[:,:,n] = np.identity(K)
    
    try :
        scale = float(scale)
        for k in range(K) :
            W[:,:,k] = pinv(A[:,:,k]) + scale * rand(N,N)
    except :
        W = rand(N,N,K)
    
    W,_,_ = iva_l(X, W_init=W, verbose=True, dispersion=dispersion)
    
    print "Joint isi between W and A is: %f"%joint_ISI(W,A)


if __name__ == "__main__" :
    scale = raw_input("Scale factor away from pseudo inverse?")
    dispersion = raw_input("Dispersion Matrix? [] for obtain from data, anything else for identity")
    main( scale, dispersion )