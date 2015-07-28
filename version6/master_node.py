import numpy as np

from set_functions import set_functions
from scipy.optimize import fmin_l_bfgs_b
from vec_mat import vec_to_mat

class master_node() :
    def __init__ (self, optimize="L-BFGS-B", verbose=False, comp_disper=50, max_iterations=20) :
        '''
        optimize : "BFGS", "L-BFGS-B", "CG"
            What optimization method to use
        
        dispersion : Int, optional
            Update dispersion matrix once every i 
            iterations. Must be greater than or equal to 20,
            else defaults to updating every iteration
        
        verbose : Bool, optional
            Print sample output? True or False
        '''
        self.optimize       = optimize
        self.comp_disper    = comp_disper
        self.verbose        = verbose
        self.max_iterations = max_iterations
    
    def fit(self, list_of_sites) :
        
        for iteration in range(self.max_iterations) :
            
            N, K, P, W, grad_and_cost, data_transfer_Y, data_transfer_grad = set_functions(list_of_sites, self.verbose, W_init=[])
            
            W, _, d = fmin_l_bfgs_b(func=grad_and_cost, x0=W, maxiter=self.comp_disper)
            
            if d['warnflag'] == 0 :
                print "converged"
                break
            else :
                continue
        
        
        if self.verbose :
            print d
        
        W_final = []
        for p in range(P) :
            W_final.append( vec_to_mat(W, N, K[p][2]) )
        
        return W_final
