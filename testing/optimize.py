import autograd.numpy as np
from autograd import grad

from autograd.numpy.linalg import inv, det, norm
from autograd.numpy.random import rand

from scipy.optimize import minimize, fmin_bfgs

from set_para import set_para
from vec_mat import vec_to_mat, mat_to_vec



def set_functions ( gradient, cost, N, K ) :
    
    def optimize_grad ( W ) :
        W = vec_to_mat(W,N,K)
        grad_value = gradient(W)
        grad_value = mat_to_vec(grad_value)
        return grad_value
    
    def optimize_cost ( W ) :
        W = vec_to_mat(W,N,K)
        cost_value = cost(W)
        return cost_value
    
    return optimize_cost, optimize_grad



def remove_mean (X) :
    N,_,K = X.shape
    for k in range(K) :
        for n in range(N) :
            X[n,:,k] = X[n,:,k] - X[n,:,k].mean()
    return X



def iva_l (X, W_init=[], verbose=False, max_iter=1024, alpha0=0.1,
           term_crit="change_in_W", term_threshold=1e-6, optimize="BFGS") :
    N,R,K = X.shape
    X = remove_mean(X)
    cost, W  = set_para(X, W_init)
    gradient = grad(cost)
    dW = gradient(W)
    optimize_cost, optimize_grad = set_functions(gradient, cost, N, K)
    
    W = mat_to_vec(W)
    if verbose :
        print "The intital value for the cost function is: ", optimize_cost(W)
        print "The intiial value for the gradient function is \n\n", optimize_grad(W)
        print "Note that the gradient is in vectorized form"
    
    W = minimize(fun=optimize_cost, x0=W, jac=optimize_grad, method=optimize)
    
    W_mat = vec_to_mat(W['x'],N,K)
    
    return W_mat, W
