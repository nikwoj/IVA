## Implementation of Laplaian Likelihood, as seen in pdf

import numpy as np
from scipy.special import gamma

def laplacian_likelihood (X, Sigma) :
    '''
    Returns likelihood funciton, given vector X and dispersion
        matrix Sigma.
        
    Inputs:
    ----------------------------------------------------------
    X = D x N matrix of column vectors
    
    '''
    
    D, N = X.shape
    
    constant = 2 ** (-d) / ( np.pi ** ((d-1) / 2.0) * np.linalg.det(Sigma) ** (1/2.0) * gamma((d+1)/2.0))
    
    ## N column vectors, return value for each one
    lap_like = np.zeros(shape=(N,1))
    
    ######
    ######
    ######
    ######
    ######
    ######
    
    ## Calculates exponential of every element in array
    exponential = np.exp()
    
    ######
    ######
    ######
    ######
    ######
    ######
    
    ## Find laplacian likelihood for every element in array
    for x in exponenetial :
        lap_like.append( constant * x )
        
    return lap_like