import numpy as np

from scipy.io import loadmat

from IVA_L import iva_l
from joint_isi import joint_ISI

def main() :
    variables_A = loadmat("SCV_IVA_case12_r001_fakeA_sqcond3_nik.mat")
    variables_S = loadmat("SCV_IVA_case12_r001.mat")
    S  = np.zeros((16, 329682, 10))
    A  = np.zeros((16, 16,     10))
    W0 = np.zeros((16, 16,     10))
    for k in range(10) :
        S [:,:,k] = variables_S['Sgt'][0,k][:,:]
        A [:,:,k] = variables_A['A'][0,k][:,:]
        W0[:,:,k] = variables_A['W0'][0,k][:,:]
    N,R,K = S.shape
    
    X = S.copy()
    
    for k in range(K) :
        X[:,:,k] = np.dot(A[:,:,k], S[:,:,k])
    
    W, d = iva_l(X, W_init=W0, verbose=True)
    
    print "Joint ISI: ", joint_ISI(W,A)
    
if __name__ == "__main__" :
    main()
    