
# coding: utf-8

# In[2]:

import numpy as np
import scipy.linalg as lin


# In[49]:

def randmv_laplace (d, T, Lambda=1, mu=[], Gamma=[]) :
    ''' Random Multivariate laplacian distrubution.
    
    
    Inputs:
    -------------------------------------------------------------------
    d = dimension of samples to generate from
    
    T = number of i.i.d. samples to generate
    
    Lambda = exponential rate parameter, real and greater than zero
    
    mu = mean vector. Must be two dimensional vector. Defaults to 
        np.zeros((d,1)). (NOTE: np.zeros(d) is different than 
        np.zeros((d,1)): one is a 1-d array, the other is a column 
        vector. One can be transposed, the other can't)
    
    Gamma = internal covariance structure. Should be d x d positive
        definite matrix such that det(Gamma) == 1, otherwise error. 
        Defaults to np.identity(d). 
        


    Outputs:
    -------------------------------------------------------------------
    Y = Matrix of vectors that satisfy the Multivariate Laplacian 
        distribution
    


    Reference:
    -------------------------------------------------------------------
    On the Multivariate Laplace Distribution
    Written by: Torbjørn Eltoft, Taesu Kim, Te-Won Lee


    '''
    
    ## Setting the parameters
    
    ## Python won't let inputs be arguements for other inputs
    if mu == [] :
        mu = np.zeros((d,1))
    elif length(mu) != d :
        raise ValueError ("mu has to be d-dimensional")
    
    
    
    ## Checks for Gamma. Note don't check for positive definiteness, just 
    ## dimensionality, det 1, and symmetry
    if Gamma == [] :
        Gamma = np.identity(d)
    elif (Gamma.shape[0] != d) or (Gamma.shape[1] != d) :
        raise ValueError ('''Gamma has to be d x d matrix''')
    elif (abs(np.linalg.det(Gamma)-1)>0.0001) :
        raise ValueError ('''Gamma must have det 1''')
    elif (np.transpose(Gamma) != Gamma) :
        raise ValueError ('''Gamma must be positive definite''')
    
    
    if Lambda < 0 :
        raise ValueError ("Lambda has to be greater than 0")
    
    
    ## Generate random "observations"
    ## ## Note: numpy, for whatever reason, seems to take row vectors, not column vectors.
    ## ## Since 
    ## ##            Y = mu + Z^.5 Gamma X 
    ## ## seems to assume column vectors, need to transpose X
    X = np.transpose(np.random.multivariate_normal(mean=np.zeros(d), cov=np.identity(d), size=(T)))
    
    Z = np.sqrt(np.random.exponential())
    
    ## Because we are allowing for T samples, need to add the mean vector to each of them. 
    ## That is all this is doing, creating T mean vectors
    mu = np.transpose(np.array([mu for x in range(T)]))
        
    Y = mu + Z * np.dot(lin.sqrtm(Gamma), X)
    
    return Y
    
    
    

