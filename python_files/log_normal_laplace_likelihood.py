
# coding: utf-8

# In[ ]:

## Implementation of Laplaian Likelihood, as seen in pdf

import numpy as np
from scipy.special import gamma

import numpy as np
import scipy.special as sp

def laplacian_likelihood (X, Sigma=[]) :
    '''
    Returns likelihood funciton, given vector X and dispersion
        matrix Sigma.
        
    Inputs:
    ----------------------------------------------------------
    X = D x N matrix of column vectors
    
    Sigma = D x D dispersion matrix. Defaults to identity
    
    
    Outputs:
    ----------------------------------------------------------
    array of values = Laplace likelihood for the N vectors
        in the X matrix
    '''
    
    
    D = X.shape[0]
    
    if Sigma==[] :
        Sigma = np.identity(D)
    elif Sigma.shape[0] != D or Sigma.shape[1] != D :
        raise ValueError ('''Sigma has to be a D x D matrix, where
                            D is the dimension of the column space
                            of X ''')
    elif np.transpose(Sigma) != Sigma :
        raise ValueError ('''Sigma has to be a symmetric matrix''')
    
    constant = 2 ** (-d) / (( np.pi ** ((d-1) / 2.0) ) * (np.linalg.det(Sigma) ** (1/2.0)) * sp.gamma((d+1)/2.0))
    

    ## Calculates exponential of every element in array
    exponential = np.exp(np.sqrt(np.sum(X * np.dot(np.pinv(Sigma), X), axis=0)))

    ## Find laplacian likelihood for every element in array
    for x in range(len(exponential)) :
        exponential[x] = exponential[x] * constant
        
    return exponential

## Log laplacian likelihood function. Obtained by taking log of normal laplacian function.


def log_laplace_like (X, Sigma=[]) :
    '''  
    Returns log of likelihood function, given vector X and dispersion
        matrix sigma. Note that Sigma defaults to identity.
        
    Inputs:
    ----------------------------------------------------------
    X = D x N matrix of column vectors
    
    Sigma = D x D dispersion matrix. Defaults to identity
    
    
    Outputs:
    ----------------------------------------------------------
    array of values = Log of laplace likelihood for the N vectors
        in the X matrix
        
    '''
    
    D = X.shape[0]
    
    if Sigma==[] :
        Sigma = np.identity(D)
    elif Sigma.shape[0] != D or Sigma.shape[1] != D :
        raise ValueError ('''Sigma has to be a D x D matrix, where
                            D is the dimension of the column space
                            of X ''')
    elif np.transpose(Sigma) != Sigma :
        raise ValueError ('''Sigma has to be a symmetric matrix''')
    
    cov = Sigma * (D+1)
    
    constant = -D * np.log(2) - (D-1)/2.0 * np.log(np.pi) - np.log(sp.gamma((D+1)/2.0)) + D/2.0 * np.log(D+1) - (1/2.0) * np.log( np.det(cov))
    
    ## Called exponential to mirror normal laplacian code.
    ## Not actually exponential, since log(exp(x))=x
    exponential = np.sqrt(D+1) * np.sqrt(np.sum(X * np.dot(np.pin(cov), X), axis=0))
    
    for x in range(len(exponential)) :
        exponential[x] = constant - exponential[x]
        
    return exponential

