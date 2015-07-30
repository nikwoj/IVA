import numpy as np

from scipy.io import loadmat

from IVA_L import iva_l
from joint_isi import joint_ISI

def main() :
    S = loadmat("d_variables.mat")['S']
    N,R,K = S.shape
    A = np.random.rand(N,N,K)
    
    X = S.copy()
    
    for k in range(K) :
        X[:,:,k] = np.dot(A[:,:,k], S[:,:,k])
    
    W, d = iva_l(X, verbose=True)
    
    print "Joint ISI: ", joint_ISI(W,A)
    
if __name__ == "__main__" :
    main()
    