import numpy as np

from sys import float_info
from numpy.linalg import pinv, det, norm
from numpy.random import rand


def set_para( X, W_init ) :
    N,R,K = X.shape
    Y = X.copy()
    YtY = np.zeros((1,R))
    dispersion = np.zeros((K,K,N))
    gradient = np.zeros((N,N,K))
    
    for n in range(N) :
        dispersion[:,:,n] = np.dot(Y[n,:,:].T, Y[n,:,:])
    
    if W_init == [] :
        W_init = np.zeros((N,N,K))
        for k in range(K) :
            W_init[:,:,k] = np.identity(N)
    

    def fit( W, dispersion ) :
        for k in range(K) :
            Y[:,:,k] = np.dot(W[:,:,k], X[:,:,k])
        
        for k in range(K) :
            YtY[:] += np.sum(Y[n,:,:].T * np.dot( pinv(dispersion[:,:,n]), Y[n,:,:].T), 0)
        
        return YtY, dispersion
    
    
    def compute_grad ( sqrt_YtY, W ) :
        for n in range(N) :
            for k in range(K) :
                value = np.dot(np.dot(pinv(dispersion[:,:,n]), Y[n,:,:].T) / sqrt_YtY,
                                         X[:,:,k].T)
                gradient[n,:,k] = value[k,n]
        return gradient
    
    return W_init, dispersion, fit, compute_grad, (N, R)



class local_node () :
    
    def __init__ (self, W_init=[]) :
        self.W_init   = W_init
    
    def initialize (self) :
        
        return self.W_init, self.disper
    
    def _fit (self, X) :
        self.W_init, self.disper, self.fit, self.compute_grad, self.shape = set_para(X, self.W_init)
    
    def initialize2 (self, W) :
        self.disper, self.compute_grad = set_para2(W) 
    
    

    
    
    


