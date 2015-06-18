## Implementation of Laplaian Likelihood, as seen in pdf

import numpy as np
from scipy.special import gamma

def laplacian_likelihood (X, Sigma=[]) :
    '''
    Returns likelihood funciton, given vector X and dispersion
        matrix Sigma.
        
    Inputs:
    ----------------------------------------------------------
    X = D x N matrix of column vectors
    
    '''
    
    
    D = X.shape[0]
    
    if Sigma==[] :
        Sigma = np.identity(D)
    
    constant = 2 ** (-d) / (( np.pi ** ((d-1) / 2.0) ) * (np.linalg.det(Sigma) ** (1/2.0)) * gamma((d+1)/2.0))
    

    ## Calculates exponential of every element in array
    exponential = np.exp(np.sqrt(np.sum(X * np.dot(Sigma, X), axis=0)))

    ## Find laplacian likelihood for every element in array
    for x in range(len(exponential)) :
        exponential[x] = exponential[x] * constant
        
    return exponential