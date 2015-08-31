import numpy as np

from numpy.linalg import pinv
from set_functions import set_functions


def set_functions ( X ) :
    N,R,K = X.shape
    Y      = np.zeros((N,N,K))
    disper = np.zeros((K,K,N))
    YdY    = np.zeros((N,R))
    
    def fit ( W ) :
        
        for k in range(K) :
            Y[:,:,k] = np.dot(W[:,:,k], X[:,:,k])
        
        for n in range(N) :
            disper[:,:,n] = np.dot(Y[n,:,:].T, Y[n,:,:])
            YdY[n,:] = Y[n,:,:].T * np.dot(pinv(disper[:,:,n]), Y[n,:,:].T)
        
        return YdY, disper
    
    return fit


class local () :
    
    def __init__ ( self, X, W_init=[] ) :
        N,R,K = X.shape
        self.fit = set_functions( X )
        
        if W_init == [] :
            W_init = np.random.rand(N,N,K)
        else :
            assert W_init.shape == (N,N,K)
    
    
    def finish ( self, W ) :
        self.W = W
    
    