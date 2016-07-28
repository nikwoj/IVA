import numpy as np
from numpy import dot, log, zeros, transpose, sum, diag
from numpy.random import rand
from numpy.linalg import det, qr
from ica import pca_whiten


class local_node() :
    '''
    Attributes:
        X = The original data (replaced by pca whitening process)
        Y = the current approximation to S, X=A*S
        W = the current unmixing matrix
    '''
    def __init__(self, X, W) :
        self.X = X
        self.W = W
    
    def avg_data(self) :
        _,_,K = self.X.shape
        Y, _ = compute_Y(self.X, self.W)
        Y = sum(Y,2) / K
        return sample(Y) 
    
    def re_order(self, order) :
        ## Permutes rows of initial unmixing matrix as defined by order
        N,_,K = self.X.shape
        permute = zeros((N,N))
        for n in range(N) :
            permute[order[n]] = 1
        
        for k in range(K) :
            self.W[:,:,k] = dot(permute, self.W[:,:,k])
    
    def initiate(self, n_components) :
        if n_components > 0 :
            self.X, wht, de_wht = whiten(self.X, n_components)
            N, R, K = self.X.shape
            return N, R, K, wht, de_wht
        else :
            return self.X.shape
    
    def local_step(self) :
        N, _, K = self.X.shape
        self.Y, YtY = compute_Y(self.X, self.W)
        w_value = 0
        for k in range(K) :
            Q, R = qr(self.W[:,:,k])
            R = diag(R)
            w_value += sum(log(abs(R)))
        return YtY, w_value
    
    def local_step2(self, sqrtYtYInv, backtrack) :
        self.gW = gradient(self.Y, self.W, sqrtYtYInv)
        self.norm = get_norm(self.gW)
        self.W_old = self.W.copy()
        self.W += self.alpha * self.gW
    
    def finish(self) :
        return self.W

def get_norm(gW) :
    K = gW.shape[2]
    return np.sum([gW[:,:,k] for k in range(K)])

def compute_Y(X, W) :
    N,R,K = X.shape
    Y = X.copy()
    for k in range(K) :
        Y[:,:,k] = dot(W[:,:,k], X[:,:,k])
    YtY = sum(Y*Y, 2)
    return Y, YtY

def sample(Y) :
    ## Extracts one in every 8 time points to get a semi-privatized
    ## part of the data
    N, R = Y.shape
    sample = zeros((N,R/8))
    for i in range(R/8) :
        sample[:, i] = Y[:, i * 8]
    
    return sample
    

def gradient(Y, W, sqrtYtYInv) :
    _, R, K = Y.shape
    dW = W.copy()
    for k in range(K) :
        phi = sqrtYtYInv * Y[:,:,k]
        dW[:,:,k] = W[:,:,k] - np.dot( np.dot(phi, transpose(Y[:,:,k]) / R), W[:,:,k])
    
    return dW

def whiten (X, n_components) :
    N,R,K = X.shape
    X_white = zeros((n_components, R, K))
    wht     = zeros((n_components, N, K))
    de_wht  = zeros((N, n_components, K))
    for k in range(K) :
        X_white[:,:,k], wht[:,:,k], de_wht = pca_whiten(X[:,:,k], n_components)
        
    return X_white, wht, de_wht

