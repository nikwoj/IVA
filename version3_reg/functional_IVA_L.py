import autograd.numpy as np
from scipy.optimize import minimize
from autograd import grad
from vec_mat import *
from stack import *

def set_parameters ( X, W_init ) :
    N,T,K    = X.shape
    print "yay?"
    def compute_cost( W ) :
        '''
        Computes the cost function for input W. Note that X is
            saved to function when iva_l is first called, and 
            so X is not a needed input
        '''
        print "yay?2"
        cost = 0
        Y    = X.copy()
        for k in range(K) :
            Y[:,:,k] = np.dot(W[:,:,k], X[:,:,k])
        
        for k in range(K) :
            cost += cost + np.log(np.abs(np.linalg.det(W[:,:,k])))
        
        cost = np.sum(np.sqrt(np.sum(Y*Y, 2))) / T - cost
        cost = cost / (N * K)
        return cost

    print "yay?3"
    gradient = grad(compute_cost)
    
    if W_init == [] :
        W = np.random.rand(N,N,K)
    else : 
        assert (W.shape[0], W.shape[2]) == (X.shape[0], X.shape[2])
        assert  W.shape[0] == W.shape[1]
        W = W_init
    W = mat_to_vec(W,K)
    
    def fit ( W ) :
        '''
        Preps the W matrix for compatibility with autograd and the 
            cost function, then returns both
        '''
        print "yay?4"
        W   = vec_to_mat(W,N,K)
        cos = compute_cost(W)
        gra = gradient(W)
        gra = mat_to_vec(gra, K)
        print  (cos, gra)
        return (cos, gra)

    return fit, W


def iva_l ( X, W_init=[], verbose=False, test=False ) :
    fit, W = set_parameters(X, W_init)
    print fit(W)
    W = minimize(fun=fit, x0=W, method="BFGS", jac=True, options={'disp':verbose})
    W = vec_to_mat(W['x'], N, K)
    
    return W