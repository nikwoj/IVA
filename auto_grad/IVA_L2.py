import autograd.numpy as np
from autograd import grad
from autograd.numpy.linalg import inv, det, norm
from autograd.numpy.random import rand


def set_para (X, W_init) :
    N,R,K = X.shape
    Y = X.copy()
    
    if W_init == [] :
        W = rand(N,N,K)
    else :
        W = W_init
    
    def comp_Y  (W) : return np.array([np.dot(W[:,:,k], X[:,:,k]) for k in range(K)])
    
    def comp_dis(Y) : return np.array([np.dot(Y[:,n,:], Y[:,n,:].T) for n in range(N)])
        
    def sqrt(Y,dispersion) :
        return np.sum(np.sqrt(np.array([np.sum(Y[:,n,:] * np.dot(inv(dispersion[n,:,:]), Y[:,n,:]), 0)
                                        for n in range(N)])))
    
    def compute_cost(W) :
        cost = 0
        Y = comp_Y(W)
        disper = comp_dis(Y)
        sqrtYtY = sqrt(Y, disper)
        for k in range(K) :
            cost = cost - np.log(np.abs(det(W[:,:,k])))
        
        cost = cost + np.sqrt(R-1) * sqrtYtY / R
        for n in range(N) :
            cost = cost + 0.5 * np.log(np.abs(det(disper[n,:,:])))
        return cost
    
    return compute_cost, W


def remove_mean (X) :
    N,_,K = X.shape
    for k in range(K) :
        for n in range(N) :
            X[n,:,k] = X[n,:,k] - X[n,:,k].mean()
    return X


def iva_l (X, W_init=[], verbose=False, max_iter=200, alpha0=0.5,
           term_crit="change_in_W", term_threshold=1e-6) :
    N,R,K = X.shape
    X = remove_mean(X)
    cost_vec = [0 for iteration in range(max_iter)]
    
    cost, W  = set_para(X, W_init)
    gradient = grad(cost)
    alpha_min = 0.1
    alpha_scale = 0.9
    
    for iteration in range(max_iter) :
        term_criterion=0
        dW = gradient(W)
        cost_vec[iteration] = cost(W)
        
        W_old = W.copy()
        W = W - alpha0 * dW
        
        ''' 
        Iteration information begins here
        '''
        if term_crit == "change_in_W" :
            for k in range(K) :
                tmp_W = W_old[:,:,k] - W[:,:,k]
                term_criterion = max(term_criterion, norm(tmp_W.reshape(N ** 2), ord=2))
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
            print "Step: %i \t W change: %f \t Cost: %f \t Gradient norm: %f"%(iteration, term_criterion, cost_vec[iteration], max([norm(dW[:,:,k], ord=2)]))
        '''
        Iteration information ends here
        For loop ends here
        '''
    
    if verbose :
        if iteration == max_iter :
            print "Algorithm may have not converged, reached max number of iterations"
        
        print " Step: %i \n W change: %f \n Cost: %f"%(iteration, term_criterion, cost_vec[iteration])
    
    return W, cost_vec