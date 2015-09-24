import numpy as np

from scipy.io import loadmat
from joint_isi import joint_ISI

from dIVA_L import dIVA_L
from local_node import local

from tests import test1, test2, test3

def main () :
    A = []
    
    W = np.zeros((16, 16,    10))
    S = np.zeros((16, 32968, 10))
    X = np.zeros((16, 32968, 10))
    
    for i in range(5) :
        ## Loop through the data sets
        variables = loadmat("SCV_IVA_case12_r00%i_fakeA_sqcond3_nik.mat" % (i+1))['A']
        for k in range(10) :
            A.append( variables[0,k] )
        
        variables = loadmat("SCV_IVA_case12_r00%i.mat" % (i+1))['Sgt']
        for k in range(10) :
            S[:,:,k] = variables[0,k]
            X[:,:,k] = np.dot(A[k][:,:], S[:,:,k])
        
        for k in range(10) :
            variables = loadmat("SCV_IVA_case12_r00%i_fakeA_sqcond3_nik.mat" % (i+1))['W0'][k,:]
            for kk in range(10) :
                W[:,:,kk] = variables[kk]
            
            test1(X, W, A, k, i)
            test2(X, W, A, k, i)
            test3(X, W, A, k, i)

   
if __name__=="__main__" :
    main()



