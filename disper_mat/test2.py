from cross_disper_mat import cross_disper_mat
from Master_disper import compute_Master_disper
from numpy.random import randn
from numpy.linalg import eigh

import numpy as np

def main() :
    A = randn(10, 100, 4)
    B = randn(10, 100, 6)
    
    
    
    #print "A is the following: \n\n", A
    #print "B is the following: \n\n", B
    
    disper_mats = []
    disper1 = np.zeros((4,4,10))
    disper2 = np.zeros((6,6,10))
    
    for n in range(10) :
        disper1[:,:,n] = np.dot(A[n,:,:].T, A[n,:,:]) / (99)
        disper2[:,:,n] = np.dot(B[n,:,:].T, B[n,:,:]) / (99)
    
    
    M = compute_Master_disper([disper1,disper2])

    for comp in range(10) :
        print M[:,:,comp]
        print eigh(M[:,:,comp])[0]
        print "\n\n"
    
    
    C = np.zeros((10, 100, 10))
    for n in range(10) :
        for k in range(2) :
            C[k,:,n] = A[n,:,k]
        for k in range(3) :
            C[k+4,:,n] = B[n,:,k]
    
    disper1 = np.zeros((5,5,10))
    for n in range(10) :
        disper1[:,:,n] = np.dot(C[:,:,n], C[:,:,n].T) / 99
    
    print disper1

if __name__ == "__main__" :
    main()