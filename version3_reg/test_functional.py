import numpy as np
from scipy.io import loadmat


from joint_isi import joint_ISI
from functional_bare_bones import iva_l

def main() :
    S = loadmat("d_variables.mat")['S']
    (N,T,K) = S.shape
    K = 2
    A = np.random.rand(N,N,K)
    X = S.copy()
    
    for k in range(K) :
        X[:,:,k] = np.dot(A[:,:,k], S[:,:,k])
    
    W = iva_l(X, verbose=True)
    
    print "Joint ISI is ", joint_ISI(W,A)

if __name__ == "__main__" :
    main()