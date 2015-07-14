import numpy as np
import scipy.optimize as opt



class Master_Node () :
    '''
    The Master Node for distributed IVA with Multivariate Laplacian
        distribution. distributed IVA has to be started by this class
    
    Methods:
    --------
    fit : 
    
    
    Attributes:
    -----------
    cost_func : Function, cost :: data -> scalar
        The desired cost function to use, defaults to _______
    
    grad_func : Function, gradient :: (data, W) -> W
        The desired gradient function to use. Options are 
        "original", ______
        
        
    verbose : Bool, optional
        Print iteration information?
        
    optimize : String, optional
        Optimize cost function with respect to which method?
        Options are "bfgs", ________
        
        "bfgs" - scipy.optimize.fmin_bfgs
    
    max_fun : Integer, optional
        Maximum number of function calls optimizer is allowed to make
        
        If optimize is "original" then sets maximum number of iterations,
        defaults to 1024
        
        If optimize is not "original", then defaults to 100* number of
        inputs to fun
    
    '''
    
    def __init__ (self, cost_func="normal", grad_func="normal", 
                  verbose=False, optimize="bfgs",  max_fun=None) :
        
        self.optimize  = optimize
        self.cost_func = cost_func
        self.grad_func = grad_func
        self.verbose   = verbose
        self.max_fun   = max_fun
    
    
    def fit (self, tot_sites) :
        '''
        Inputs:
        -------
        tot_sites : dictionary of objects (or prototcols to reach
            site)
        
        Outputs:
        --------
        Just gives every site the W matrices that solve x = As
        '''

        self.tot_sites = tot_sites
        subjects       = []
        shape          = (0, 0)
        W_init         = []
        self.sqrtYtY   = 0
        
        for site in self.tot_sites :
            ## Gonna need to find out how to properly do this data transmission            
            W_i_site = self.tot_sites[site].parameters (
                cost_func=self.cost_func, grad_func=self.grad_func, 
                verbose=self.verbose, optimize=self.optimize
            )
            
            subjects.append(len(self.tot_sites[site].W_init))
            shape = self.tot_sites[site].W_init[0].shape
            
            for W in W_i_site :
                W_init.append(W)
            
        
        self.shape = shape
        self.grad  = grad = [np.zeros(shape=(self.shape[0], self.shape[0])) 
                                            for p in range(len(subjects)) 
                                            for x in range(subjects[p]) 
                                            ]
        
        W = optimize(self._cost_function, W_init, self._grad_function, self.optimize, self.verbose)['x']
        
        cc = 0
        for site in self.tot_sites :
            ## Gonna need to find out how to properly do this data transmission
            for i in range(subjects[cc]) : 
                self.tot_sites[site].fit(W)
            cc += 1
    
    
    def _cost_function (self, W, cost_func) :
        '''
        The cost function used, needs Y^2 values,
        '''
        
        if self.verbose :
            print "Running cost function"
        
        YtY = np.zeros(shape=(self.N, self.T))
        for site in self.tot_sites :
            ## Gonna need to find out how to properly do this data transmission
            YtY += tot_sites[site].fit(W, cost_val=True)
        
        self.sqrtYtY = np.sqrt(YtY)
        
        return self._compute_cost(W)
    
    
    def _compute_cost (self, W) :
        '''
        Computes the current value of the cost function
        '''
        
        N, T = self.shape
        K = len(W)
        current_cost = 0
        
        for k in range(K) :
            current_cost += np.log(abs(np.linalg.det(W[k])))
        
        current_cost = np.sum(np.sum(self.sqrtYtY)) / T - current_cost
        current_cost = current_cost / (N*K)
        
        return current_cost
    
    
    def _grad_function (self, W) :
        '''
        The gradient used, needs data, so always computed locally
            and sent to master node.
            
            When optimize function is "original", then this is 
            never called, as gradient is computed and used locally
        '''
        
        if self.verbose :
            print "Running gradient function"
        
        # YtY = 0
        # for site in self.tot_sites :
        #     YtY += site.fit(W, cost_val=True)
        
        cc = 0
        for site in self.tot_sites :
            grad_site = self.tot_sites[site].fit(W, self.sqrtYtY, grad_val=True)
            
            for k in range(grad_site) :
                ## Gonna need to find out how to properly do this data transmission
                grad[cc+k][:,:] = grad_site[k][:,:]
                
            cc += len(grad_site)
        
        print grad
        
        return grad
    
    
def optimize (cost, init_guess, grad, optimize, verbose=False) :
    '''
    
    '''
    
    if verbose :
        print "Running optimization function"
    
    if optimize == "bfgs" :
        W = opt.fmin_bfgs(f=cost, x0=init_guess, fprime=grad)
    
    return W
    