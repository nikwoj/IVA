import numpy as np


def stack (W) : 
    ## Takes matrix and outputs vector
    return W.reshape(W.shape[0]**2)

def unstack(W) :
    ## Takes vector and outputs matrix stacked according to stack
    W = np.array(W)
    N = np.sqrt(W.shape)
    return W.reshape(N,N)
