import autograd.numpy as np
import sys
import scipy.optimize as opt

from autograd.numpy.linalg import pinv
from autograd import grad
from square_root import *
from vec_mat import *
from stack   import *
## Two functions contained in here: stack and unstack.
## Stack takes matrice and turns it into vector, unstack does inverse
## Only works fro two dimensional arrays

class IVA_L () :
    '''
    
    '''
    
    def __init__ (self, verbose=False, W_init=[]) :
        self.verbose = verbose
        self.W_init  = W_init
        self.grad    = grad(_compute_cost)
    
    
    def fit(self, X) :
        '''
        Fits the IVA problem 
        '''
        N,T,K = X.shape
        
        if self.W_init == [] :
            # self.W_init = np.random.rand(N,N,K)
            self.W_init = np.zeros((N,N,K))
            for k in range(K) :
                self.W_init = np.identity(N)
        
        assert (self.W_init.shape[0], self.W_init.shape[2]) == (N,K)
        self.X = X
        
        print _compute_cost(self.W_init)
        
        W = mat_to_vec(self.W_init, K)
        
        
        
        # W = opt.fmin_bfgs(f=self._compute_cost, x0=W, fprime=self._grad_function)
        W = opt.minimize(fun=self.function, x0=W, method="BFGS", jac=True, options={'disp':True})
        
        print W
        
        W = vec_to_mat (W['x'], N, K)
        self.W = W
    
    
    
    
    
    
    def _cost_function (self, W) : self.function(W, cost_val=True)
    
    def _grad_function (self, W) : self.function(W, grad_val=True)
    
    
    
    
    
    def function (self, W, cost_val=False, grad_val=False) :
        '''
        Prepares a matrix to be sent to either cost function or gradient
            function.
        '''
        N, N, K = self.W_init.shape
        W = vec_to_mat(W,N,K)
        cost = _compute_cost(W) 
        print "doing grad"
        g = self._auto_grad(W)
        return (cost, grad)
    
    
    
    
    
    
    
    
    
    def _auto_grad(self, W) :
        '''
        Computes the change matrix dW, that tells W how and where to change.
            Takes in list of 2-D arrays, outputs gradient vectorized
        '''
        
        W = self.grad(W)
        
        return mat_to_vec(W)
        
    
    
    
    
    
    """
    def _grad_function (self, W) :
        '''
        Computes the change matrix dW, that tells W how and where to change
        
        Inputs:
        -------
        W: The unmixing matrix
        
        Y: The current data
        
        sqrtYtYInv: Summary of the current Y across sites 
        
        Outputs:
        --------
        dW: The matrix which tells W how much and in what direction to 
            change
        '''
        
        N, T, K = self.X.shape
        Y  = self.X.copy()
        dW = W.copy()
        
        for k in range(K) :
            Y[:,:,k] = np.dot(W[:,:,k], self.X[:,:,k])
        
        sqrtYtYInv = 1 / (np.sqrt(np.sum(Y*Y,2)) + sys.float_info.epsilon)
        
        for k in range(K) :
            phi = sqrtYtYInv * Y[:,:,k]
            dW[:,:,k] = pinv(W[:,:,k]).T - np.dot( np.dot(phi, np.transpose(Y[:,:,k]) / T), pinv(W[:,:,k])).T
        
        dW = mat_to_vec(dW, K)
        
        return dW
    
    """
    
    
    

def _compute_cost(W) :
    '''
    Outputs the cost of the current iteration
    
    Inputs:
    -------
    W: The unmixing matrix
    
    Outputs:
    --------
    current_cost: A number associated to the cost of the current iteration.
    '''
    N, T, K = W.shape
    current_cost = 0
    
    
    
    # print W[0,0,0].cast()
    # print [method for method in dir(W[0,0,0]) if callable(getattr(W[0,0,0], method))]
    # YtY = []
    
    # for k in range(K) :
    #     # Y[:,:,k] = np.dot(W[:,:,k], self.X[:,:,k])
    #     YtY.append( np.dot(W[:,:,k], self.X[:,:,k]) * np.dot(W[:,:,k], self.X[:,:,k]) )
    
    # YtY     = sum(YtY)
    # sqrtYtY = square_root(YtY, 1e-8)
    # ## W is an ndarray with dtype=object during autograd, which np.sqrt cannot understand
    # ## Need to take root, so just approximate it
    
    # for k in range(K) :
    #     current_cost = current_cost + np.log(np.abs(np.linalg.det(W[:,:,k])))
    
    # current_cost = (-1)*current_cost + np.sum(sqrtYtY) / T
    # current_cost = current_cost / (N*K)
    
    # return current_cost
    
    for k in range(K) :
        print W[:,:,k]
        current_cost += np.linalg.det(W[:,:,k])
    return current_cost

    
    
    