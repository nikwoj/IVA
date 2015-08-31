import numpy as np

from vec_mat import vec_to_mat, mat_to_vec

from set_para import set_para



class dIVA_L () :
    '''
    The distirbuted IVA class
    
    Initializers : all optional
    ------------
    verbose : bool, True or False
        Display some output?
    
    bounds : Tuple
    
    m : 
    
    factr : 
    
    pgtol :  
    
    epsilon : 
    
    iprint : 
    
    maxfun : 
    
    maxiter : 
    
    disp : 
    
    callback : 
    
    Methods
    -------
    fit : 
    '''
    
    def __init__( 
        self, verbose=False, bounds=None, m=10, factr=10000000.0, 
        pgtol=1e-05, epsilon=1e-08, iprint=-1, maxfun=15000, 
        maxiter=15000, disp=None, callback=None
        ) :
        
        self.verbose  = verbose
        self.bounds   = bounds
        self.m        = m
        self.factr    = factr
        self.pgtol    = pgtol
        self.epsilon  = epsilon
        self.iprint   = iprint
        self.maxfun   = maxfun 
        self.maxiter  = maxiter
        self.disp     = disp
        self.callback = callback
    
    def fit( self, list_of_sites ) :
        '''
        Fit the model with data from the sites.
        '''
        cost_and_grad, K, N, R, KK, W = set_para( 
            list_of_sites, self.verbose
            )
        
        self.W, self.i, self.d = l_bfgs_b(
            func=cost_and_grad, x0=W, bounds=self.bounds, m=self.m, 
            factr=self.factr, pgtol=self.pgtol, epsilon=self.epsilon,
            iprint=self.iprint, maxfun=self.maxfun, maxiter=self.maxiter,
            disp=self.disp, callback=self.callback
            )
        
        if self.verbose :
            print "Finished optimization, making matrix and sending to sites now."
            print "Iteration information:"
            print d
        
        W = vec_to_mat( W, N, KK )
        
        
        count = 0
        tot   = 0
        for k in K :
            site.finish( W[:,:, tot:tot+k ], i, d )
            
            tot += k
    
    
    
    
    
    