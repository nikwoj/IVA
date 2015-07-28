from cross_disper_mat import cross_disper_mat
from Master_disper import compute_Master_disper
from numpy.random import randn
from numpy.linalg import eigh

import numpy as np

def main() :
    
    np.set_printoptions(precision=2)

    A = randn(10, 10000, 5)
    B = randn(10, 10000, 5)
    
    
    
    #print "A is the following: \n\n", A
    #print "B is the following: \n\n", B
    
    disper_mats = []
    disper1 = np.zeros((5,5,10))
    disper2 = np.zeros((5,5,10))
    
    for n in range(10) :
        disper1[:,:,n] = np.dot(A[n,:,:].T, A[n,:,:]) / (9999)
        disper2[:,:,n] = np.dot(B[n,:,:].T, B[n,:,:]) / (9999)
        # disper1[:,:,n] = np.identity(2) * 2
        # disper2[:,:,n] = np.identity(5) * 5
    
    M = compute_Master_disper([disper1,disper2])

    for comp in range(10) :
        print eigh(M[:,:,comp])[0]
    
    
    C = np.zeros((10, 10000, 10))
    for n in range(10) :
        for k in range(5) :
            C[k,:,n] = A[n,:,k]
        for k in range(5) :
            C[k+5,:,n] = B[n,:,k]
    
    disper1 = np.zeros((10,10,10))
    for n in range(10) :
        disper1[:,:,n] = np.dot(C[:,:,n], C[:,:,n].T) / 9999
        # print np.dot(C[:,:,n], C[:,:,n].T) / 99
        print disper1[:,:,n] - M[:,:,n]

if __name__ == "__main__" :
    main()