import autograd.numpy as np
from scipy.optimize import minimize
from autograd import value_and_grad


from vec_mat import *
from stack import *



def set_para (X) :
    
    def compute_cost(W) :
        print "yay?"
        cost = 0
        Y    = X.copy()
        for k in range(K) :
            Y[:,:,k] = np.dot(W[:,:,k], X[:,:,k])
        
        for k in range(K) :
            cost += cost + np.log(np.abs(np.linalg.det(W[:,:,k])))
        
        cost = np.sum(np.sqrt(np.sum(Y*Y, 2))) / T - cost
        cost = cost / (N * K)
        return cost
    
    print "yay2"
    return compute_cost



def iva_l ( X, W_init=[], verbose=False) :
    
    cost = set_para (X)
    grad = grad(cost)
    
    N,T,K = X.shape
    if W_init==[] :
        W = np.random.rand(N,N,K)
    else :
        W = W_init

    print cost_and_grad(W)
    
    W = minimize(fun=cost_and_grad, x0=W, args=(X), method="BFGS", jac=gradient, options={'disp':verbose})
    W = vec_to_mat(W['x'], N, K)
    
    return W