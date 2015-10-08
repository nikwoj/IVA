import numpy as np

from scipy.optimize import fmin_l_bfgs_b
from numpy.linalg import pinv, det

from vec_mat import mat_to_vec, vec_to_mat


def set_functions ( X, W_init=[], verbose=False ) :
    N,R,K = X.shape
    Y = X.copy()
    
    if W_init == [] :
        W_init = np.random.rand(N,N,K)
    
    
    W_master = np.zeros((N,N,K))
    gradient = np.zeros((N,N,K))
    disper   = np.zeros((K,K,N))
    A        = np.zeros((K,R,N))
    sqrt_YtY = np.zeros((N,R))
    
    def cost_and_grad ( W ) :
        if verbose:
            print "Running cost and gradient function"
        
        W_master = vec_to_mat(W,N,K)
        for k in range(K) :
            Y[:,:,k] = np.dot(W_master[:,:,k], X[:,:,k])
        
        for n in range(N) :
            disper[:,:,n] = np.dot(Y[n,:,:].T, Y[n,:,:])
            
            A[:,:,n]      = np.dot(pinv(disper[:,:,n]), Y[n,:,:].T)
            sqrt_YtY[n,:] = np.sqrt( np.sum( Y[n,:,:].T * A[:,:,n], 0 ) )
        
        ## Computing cost
        cost = np.sum( sqrt_YtY ) * (np.sqrt(R-1) / R)
        for n in range(N) :
            cost += .5 * np.log(det(disper[:,:,n]))
        
        for k in range(K) :
            cost = cost - np.log(np.abs(det(W_master[:,:,k])))
        
        ## Computing Gradient
        gradient = np.zeros((N,N,K))
        for n in range(N) :
            B = A[:,:,n] * ( ( np.sqrt(R-1) / R ) / sqrt_YtY[n,:])
            C = np.identity(K) - np.dot(B, Y[n,:,:])
            value = B + np.dot(C, A[:,:,n])
            
            for k in range(K) :
                gradient[n,:,k] = np.dot(value[k,:], X[:,:,k].T)
        
        for k in range(K) :
            gradient[:,:,k] = gradient[:,:,k] - pinv(W_master[:,:,k]).T
        
        
        gradient = mat_to_vec(gradient)
        
        if verbose :
            print cost
        
        return cost, gradient
    
    return cost_and_grad, W_init


def remove_mean (X) :
    N,_,K = X.shape
    for k in range(K) :
        for n in range(N) :
            X[n,:,k] = X[n,:,k] - X[n,:,k].mean()
    return X


def iva_l (X, W_init=[], verbose=False) :
    N,R,K = X.shape
    X = remove_mean(X)
    cost_and_grad, W = set_functions(X, W_init, verbose)
    W = mat_to_vec(W)
    
    W, _, d = fmin_l_bfgs_b(func=cost_and_grad, x0=W)
    
    if verbose :
        print d
    W = vec_to_mat(W, N, K)
    
    return W, d
