import numpy as np

from sys import float_info
from numpy.linalg import pinv, det, norm
from numpy.random import rand


def _get_dispersion(X) :
    '''
    
    '''
    N,_,K  = X.shape
    disper = np.zeros((K, K, N))
    for n in range(N) :
        value = np.dot(X[n,:,:].T, X[n,:,:]) / (N+1)
        disper[:,:,n] = value
    
    return disper


def _set_para(X, W_init, dispersion) :
    N,R,K = X.shape

    if W_init == [] :
        W = rand(N,N,K)
    else :
        W = W_init

    Y  = X.copy()
    dW = W.copy()
    

    def Y_and_dW(W) :
        '''
        Computes Y, dW, and dispersion matrix.
        Reutrns sqrtYtY, dW
        '''
        YtY = np.zeros((N,R))
        for k in range(K) :
            Y[:,:,k] = np.dot(W[:,:,k], X[:,:,k])
        
        disper = _get_dispersion(Y)
        
        for n in range(N) :
            YtY[n,:] = np.sum(Y[n,:,:].T * np.dot(pinv(disper[:,:,n]), Y[n,:,:].T), 0)
        
        sqrt_YtY = np.sqrt(YtY)
        sqrt_YtY_inv = 1 / (sqrt_YtY + float_info.epsilon)

        for k in range(K) :
            phi = np.dot(sqrt_YtY_inv * Y[:,:,k], Y[:,:,k].T) / R
            dW[:,:,k] = pinv(W[:,:,k]).T - np.dot(phi, pinv(W[:,:,k]).T)
        
        return sqrt_YtY, dW
    

    def compute_cost(W, sqrt_YtY) :
        cost = 0
        for k in range(K) :
            cost = cost + np.log(np.abs(det(W[:,:,k])))
        
        cost = np.sum(sqrt_YtY) / R - cost
        cost = cost / (N * K)
        
        return cost
    
    return compute_cost, Y_and_dW, W



def iva_l (X, W_init=[], dispersion=[], verbose=False, max_iter=1024,
           alpha0=0.1, term_threshold=1e-6, term_crit="change_in_cost",
           iter_get_disper=5) :
    '''
    
    '''
    K = X.shape[2]
    alpha_min   = 0.1
    alpha_scale = 0.9
    cost_vec    = [np.NaN for iteration in range(max_iter)]
    
    compute_cost, Y_and_dW, W = _set_para(X, W_init, dispersion)
    
    for iteration in range(max_iter) :
        term_criterion = 0
        sqrt_YtY, dW = Y_and_dW(W)
        
        W_old = W.copy()
        W = W + alpha0 * dW
        
        cost_vec[iteration] = compute_cost(W, sqrt_YtY)
        
        ''' 
        Iteration information begins here
        '''
        if term_crit == "change_in_W" :
            for k in range(K) :
                tmp_W = W_old[:,:,k] - W[:,:,k]
                term_criterion = max(term_criterion, norm(tmp_W[:,:], ord=2))
        elif term_crit == "change_in_cost" :
            if iteration == 1 :
                term_criterion = 1
            else :
                term_criterion = ( abs(cost_vec[iteration] - cost_vec[iteration-1])
                                   / abs(cost_vec[iteration]) )
        
        if term_criterion < term_threshold or iteration == max_iter :
            break
        elif np.isinf(cost_vec[iteration]) :
            if verbose :
                print "W blew up, restarting with new intital value"
            for k in range(K) :
                W[:,:,k] = np.identity(N) + 0.1 * rand(N,N)
        elif iteration > 1 and cost_vec[iteration] > cost_vec[iteration-1] :
            alpha0 = max([alpha_min, alpha_scale * alpha0])
        
        if verbose :
            print "Step: %i \t W change: %f \t Cost: %f"%(iteration, term_criterion, cost_vec[iteration])
        '''
        Iteration information ends here
        For loop ends here
        '''
    
    if verbose :
        if iteration == max_iter :
            print "Algorithm may have not converged, reached max number of iterations"
        
        print " Step: %i \n W change: %f \n Cost: %f"%(iteration, term_criterion, cost_vec[iteration])
    
    return W, iteration, cost_vec


# class IVA_L () :
    
#     def __init__ (self, alpha0, term_threshold, term_crit, max_iter, W_init, dispersion) :
#         self.alpha0         = alpha0
#         self.term_threshold = term_threshold
#         self.term_crit      = term_crit
#         self.max_iter       = max_iter
#         self.W_init         = W_init
#         self.dispersion     = dispersion
    
#     def fit (self, X) :