import numpy as np
from numpy import dot, log, zeros, transpose, sum
from numpy.random import rand
from numpy.linalg import det
from ica import pca_whiten

def compute_Y(X, W) :
    N,R,K = X.shape
    Y = X.copy()
    for k in range(K) :
        Y[:,:,k] = dot(W[:,:,k], X[:,:,k])
    YtY = sum(Y*Y, 2)
    return Y, YtY

def gradient(Y, W, sqrtYtYInv) :
    _, T, K = Y.shape
    dW = W.copy()
    for k in range(K) :
        phi = sqrtYtYInv * Y[:,:,k]
        dW[:,:,k] = W[:,:,k] - np.dot( np.dot(phi, transpose(Y[:,:,k]) / T), W[:,:,k])
    
    return dW

def whiten (X, n_components) :
    N,T,K = X.shape
    X_white = zeros((n_components, T, K))
    wht     = zeros((n_components, N, K))
    de_wht  = zeros((N, n_components, K))
    for k in range(K) :
        X_white[:,:,k], wht[:,:,k], de_wht = pca_whiten(X[:,:,k], n_components)
    
    return X_white, wht, de_wht


class local_node() :
    '''
    Attributes:
        X = The original data (replaced by pca whitening process)
        Y = the current approximation to S, X=A*S
        W = the current unmixing matrix
    '''
    def __init__(self, X) :
        self.X = X
    
    def node_step(self, sqrtYtYInv, al0) :
        _, _, K = self.X.shape
        dW = gradient(self.Y, self.W, sqrtYtYInv)
        self.W += al0 * gradient(self.Y, self.W, sqrtYtYInv)
        self.Y, YtY = compute_Y(self.X, self.W)
        w_value = sum([log(abs(det(self.W[:,:,k]))) for k in range(K)])
        return YtY, w_value
    
    def initiate(self, n_components, W=[]) :
        self.X, wht, de_wht = whiten(self.X, n_components)
        N, T, K = self.X.shape
        if W == [] :
            self.W = rand(N,N,K)
        else :
            self.W = W
        
        self.Y, YtY = compute_Y(self.X, self.W)
        return YtY, K, wht, de_wht
    
    def finish(self) :
        return self.W
