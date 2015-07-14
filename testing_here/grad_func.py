import numpy as np


def original (W, Y, sqrtYtYInv) :
    '''
    Computes the change matrix dW. 
    
    A local function, should be computed at each individual site, as
        involves site specific data (which shouldn't be sent to other
        sites)
    
    Inputs:
    -------
    W : the unmixing matrix
    
    phi : A 4-D matrix, made of two dimensional matrices for each
        subject for each site.
        
        phi = np.dot(sqrtYtYInv * Y[:,:,k,p], np.transpose(Y[:,:,k,p]))
        
    Outputs:
    --------
    dW : Change matrix
    '''
    P    = len(Y)
    N, T = Y[0][0].shape
    
    dW = [[np.random.rand(N,N) for x in range(len(Y[p]))] for p in range(P)]

    
    for p in range(P) :
        for k in range(len(Y[p])) :
            phi = np.dot(sqrtYtYInv * Y[p][k], np.transpose(Y[p][k][:,:])) / T
            dW[p][k] = W[p][k] - np.dot(phi, W[p][k])
    
    return dW