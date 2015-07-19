import numpy as np
import grad_func as grad

class local_node () :
    '''
    
    Inputs:
    -------
    W_init : List of 2-D arrays, optional
        Initial guess for W matrices 
    
    X : List of 2-D arrays
        The data to be used for this site in IVA
    '''
    
    def __init__ (self, X, W_init=[]) :

        self.X      = X
        
        if W_init == [] :
            self.W_init = [np.identity(X[0].shape[0]) for p in range(len(X))]
        else :
            self.W_init = W_init
    
    
    def parameters (self, cost_func, grad_func, verbose,
                    optimize) :
        '''
        Parameters used in start of iteration, given by Master Node
        
        In return, Master Node recieves initial estimates for W from
            each site
        '''
        self.cost_func = cost_func
        self.grad_func = grad_func
        self.verbose   = verbose
        self.optimize  = optimize
        
        return self.W_init
    
    
    def fit (self, W, sqrtYtY=0, func_val=False, grad_val=False) :
        '''
        Updates Y^2 or gradient after recieving new W
            matrix, and sends it to master node.
        
        Inputs:
        -------
        W : the current mixing matrix, sent from master node
        '''
        
        K = len(W)
        print W[0].shape
        Y = [self.X[0].copy() for k in range(len(self.X))]
        
        for k in range(K) :
            Y[k][:,:] = np.dot(W[k][:,:], self.X[k][:,:])
        
        if func_val :
            YtY = np.sum(Y*Y, 2)
            return YtY
        
        elif grad_val :
            return grad_func(W, Y, sqrtYtY)
        
        else :
            self.W = W
    
    
    def grad_func (self, Y, sqrtYtY) :
        '''
        The gradient function used in the optimization.
        '''
        
        if grad_func == "original" :
            return grad.original(W, Y, sqrtYtY)
        