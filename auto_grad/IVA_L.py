import autograd.numpy as np
from autograd import grad

from autograd.numpy.linalg import pinv, det


def _get_dispersion(X) :
    '''
    
    '''
    N,_,K  = X.shape
    disper = np.zeros((K, K, N))
    for n in range(N) :
        value = np.dot(X[n,:,:].T, X[n,:,:]) / (N+1)
        disper[:,:,n] = value
    
    return disper




def set_para (X, W_init) :
    N,R,K = X.shape
    Y = X.copy()
    
    if W_init == [] :
        W = rand(N,N,K)
    else :
        W = W_init
    
    def compute_cost(W) :
        for k in range(K) :
            Y[:,:,k] = np.dot(W[:,:,k], X[:,:,k])
        
        disper = _get_dispersion(Y)
        cost   = 0
        YtY    = np.zeros((N,R))
        for n in range(N) :
            YtY[n,:] = np.sum(Y[n,:,:].T * np.dot(pinv(disper[:,:,n]), Y[n,:,:].T), 0)
        
        for k in range(K) :
            cost += cost + np.log(np.abs(np.linalg.det(W[:,:,k])))
        
        sqrtYtY = np.sum(np.sqrt(sum(YtY)))
        cost = sqrtYtY / R - cost
        cost = cost / (N * K)

        return cost
    
    return compute_cost, W


def iva_l (X, W_init=[], verbose=False, max_iter=1024, alpha0=0.1, term_crit="change_in_cost" ) :
    N,R,K = X.shape
    cost_vec = []
    alpha_min = 0.1
    alpha_scale = 0.9
    
    compute_cost, W = set_para(X, W_init)
    gradient = grad(compute_cost)
    
    for iteration in range(max_iter) :
        term_criterion = 0
        
        dW = gradient(W)
        cost_vec.append(compute_cost(W))
        
        W_old = W.copy()
        W = W + alpha0 * dW
        
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
    
    return W, cost_vec



















# def main() :
#     X = np.random.rand(10,100,2)
    
#     print "Defining cost function"
#     cost = set_para(X)
    
#     print "Defining grad function"
#     gradient = grad(cost)
    
    
#     print "Does cost function work? Testing: "
#     try:
#         print cost(np.random.rand(10,10,2))
#     except :
#         print "Cost function doesn't work"
    
    
#     print "Does gradient work? Testing: "
#     print gradient(np.random.rand(10,10,2))

# if __name__ == "__main__" :
#     main()