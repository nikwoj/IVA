import numpy as np
from numpy.linalg import eigh

def cross_disper_mat(disper1, disper2) :
    '''
    Both disper1 and disper2 should be 2-D pos definite
        matrices. cross_disper_mat computes what the off
        diagonal entry in the master dispersion matrix 
        should be by taking the eigenvalues of each of 
        disper1 and disper2 and computing the associated
        sum of outer products. The resulting matrix will 
        be off full rank.
        
        Shape of resulting matrix: rows = rows of disper1
                                   cols = cols of disper2
    '''
    eig1  = eigh(disper1)
    eig2  = eigh(disper2)
    c_mat = np.zeros((disper1.shape[0], disper2.shape[0]))
    
    for e1 in range(len(eig1[0])) :
        for e2 in range(len(eig2[0])) :
            c_mat += np.sqrt(eig1[0][e1] * eig2[0][e2]) * np.outer(eig1[1][:,e1], eig2[1][:,e2])
    
    return c_mat / (len(eig1[0]) * len(eig2[0]))
    # return c_mat