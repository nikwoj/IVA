## Just general need modules
import numpy as np
# from sklearn.preprocessing import normalize
import sys

## for testing purposes, data was generated in MATLAB
## Will create (or rather, fix...) generating function at later date
import scipy.io as sio


def _vecnorm(A) : return normalize(A, axis=0, norm='l2')
''' Note: Only used when test_IVA_L is called in order to orthonomalize mixing matrix '''


def _get_dW (W, Y, sqrtYtYInv) :
    '''
    Computes the change matrix dW. 
    
    A local function, should be computed at each individual site, as
        involves site specific data (which shouldn't be sent to other
        sites)
    
    Inputs:
    -------
    W : the unmixing matrix
    
    phi : A 4-D matrix, made of two dimensional matrices for each
        subject for each site.
        
        phi = np.dot(sqrtYtYInv * Y[:,:,k,p], np.transpose(Y[:,:,k,p]))
        
    Outputs:
    --------
    dW : Change matrix
    '''
    P    = len(Y)
    N, T = Y[0][0].shape
    
    dW = [[np.random.rand(N,N) for x in range(len(Y[p]))] for p in range(P)]

    
    for p in range(P) :
        for k in range(len(Y[p])) :
            phi = np.dot(sqrtYtYInv * Y[p][k], np.transpose(Y[p][k][:,:])) / T
            dW[p][k] = W[p][k] - np.dot(phi, W[p][k])
    
    return dW


def _compute_cost(W, sqrtYtY) :
    '''
    Outputs the cost of the current iteration
    
    A local function, could be changed without changing functionality to
        allow it to be computed at every site. Computing at Master Node
        saves overall time, however, as each site should get the same 
        value.
    
    Inputs:
    -------
    W: The unmixing matrix
    
    sqrtYtYInv: Summary of the current Y across sites
    
    Outputs:
    --------
    current_cost: A number associated to the cost of the current iteration.
    '''
    
    P = len(W)
    N, T = sqrtYtY.shape
    K = sum([len(W[p]) for p in range(P)])
    current_cost = 0
    
    for p in range(P) :
        for k in range(len(W[p])) :
            current_cost = current_cost + np.log(abs(np.linalg.det(W[p][k])))
    
    current_cost = np.sum(np.sum(sqrtYtY,1)) / T - current_cost
    current_cost = current_cost / (N*K)
    
    return current_cost



def diva_l (X, alpha0=0.1, term_threshold=1e-6, term_crit='ChangeInCost',
           max_iter=1024, W_init=[], verbose=False) :
    '''
    IVA_L is the Independent Vector Analysis using multivariate Laplacian 
        distribution
    
    Inputs:
    -------
    X : List of lists of 2-D arrays
        Has total list, length of which is total number of sites, each 
        element of which is a list of subjects within that site. Every 
        subject is a 2-D array, shape of which is (row, column).
    
    alpha0 : Float, optional
        Learning rate. Defaults to 0.1
    
    term_threshold : Float, optional
        How low does the termination Criterion have to be in order for the 
        algorithm to stop? Defaults to 1e-6
        
    term_crit : String, optional
        Termination Criterion. Only two options: ChangeInCost and ChangeInW. 
        Defaults to ChangeInCost.
        
    max_iter : Int, optional
        The maximum number of iterations the algorithm is allowed to run for.
        Defaults to 1024.
        
    W_init : array, shape=(K,N,N), optional
        Initial guess for W? Defaults to np.random.rand(N,N,K), where N
        is number of rows of X, and K is number of rows deep X is 
        
        (ie X.shape = (N,T,K), and W.shape = (N,N,K))
        
    verbose : Bool, optional
        Print iteration information to output? Defaults to False
    
    Outputs:
    --------
    W : list of arrays, shape = (N,N)
        The unmixing matrix W for each subject, ordered in order
        that subjects were recieved in
    
    iteration : 
    '''
    

    P = len(X)
    N,T = X[0][0][:,:].shape
    cost = [np.NaN for x in range(max_iter)]
    
    ## Make Y have same dimension as X. Equivalent to X.copy(), but faster    
    Y = [[np.zeros(shape=Z.shape) for Z in X[p]] for p in range(P)]
    alpha_min   = 0.1
    alpha_scale = 0.9
    
    if W_init == [] :
        W = [[np.random.rand(N, N) for x in range(len(X[p]))] for p in range(P)]
    else :
        W = W_init
    
    if (term_crit != 'ChangeInW') and (term_crit != 'ChangeInCost') :
        raise ValueError ('''term_crit has to be either 
                          'ChangeInW' or 'ChangeInCost' ''')
    
    ## Main Loop
    for iteration in range(max_iter) :
        term_criterion = 0
        
        YtY = np.zeros(shape=(N,T,P))
        
        ## Happens Locally
        for p in range(P) :
            for k in range(len(X[p])) :
                Y[p][k]     = np.dot(W[p][k][:,:], X[p][k][:,:])
                YtY[:,:,p] += Y[p][k][:,:] * Y[p][k][:,:]
        
        ## Happens at Master Node
        sqrtYtY    = np.sqrt(np.sum(YtY, 2))
        sqrtYtYInv = 1 / (sqrtYtY + sys.float_info.epsilon)
        
        W_old = [[np.random.rand(Z.shape[0], Z.shape[0]) for Z in X[p]] for p in range(P)]
        ## Creating W_old, but have to do it step by step
        for p in range(P) :
            for k in range(len(X[p])) :
                W_old[p][k][:,:] = W[p][k][:,:]
        
        dW = _get_dW(W,Y,sqrtYtYInv)
        
        for p in range(P) :
            for k in range(len(X[p])) :
                W[p][k][:,:] += alpha0 * dW[p][k][:,:]
        
        cost[iteration] = _compute_cost(W, sqrtYtY)
        
        ## Check termination Criterion
        if term_crit == 'ChangeInW' :
            for p in range(P) :
                for k in range(len(X[p])) :
                    tmp_W = W_old[p][k][:,:] - W[p][k][:,:]
                    term_criterion = max(term_criterion, np.linalg.norm(tmp_W[:,:], ord=2))
        
        elif term_crit == 'ChangeInCost' :
            if iteration == 1 :
                term_criterion = 1
            
            else :
                term_criterion = (abs(cost[iteration-1]-cost[iteration])
                                 / abs(cost[iteration]))
        
        
        ## Check termination condition
        if term_criterion < term_threshold or iteration == max_iter :
            break
        elif np.isinf(cost[iteration]) :
            if verbose :
                print "W blew up, restarting with new initial value"
            for p in range(P) :
                for k in range(len(X[p])) :
                    W[:,:,k] = np.identity(N) + 0.1 * np.random.rand(N)
        
        elif iteration > 1 and cost[iteration] > cost[iteration-1] :
            alpha0 = max([alpha_min, alpha_scale * alpha0])
        
        ## Display iteration information
        if verbose :
            print "Step: %i \t W change: %f \t Cost %f" % (iteration, term_criterion, cost[iteration])
    
    ## End iteration
    
    if iteration==max_iter :
        print ('''Algorithm may have not converged, reached max
               number of iterations ''')
    
    ## Finish display
    if verbose :
        print "Algorithim converged, end results are: "
        print " Step: %i \n W change: %f \n Cost %f \n\n" % (iteration, term_criterion, cost[iteration])
    
    return W, iteration, cost



class dIVA_L() :
    '''
    The class for dIVA_L, in a similar vein to the class for IVA_L
        or PCA or ICA
    '''
    
    def __init__(self, alpha0=0.1, term_threshold=1e-6, 
                 term_crit="ChangeInCost", max_iter=1024, W_init=[]) :
        self.alpha0         = alpha0
        self.term_threshold = term_threshold
        self.term_crit      = term_crit
        self.max_iter       = max_iter1024
        self.W_init         = W_init
        self.verbose        = verbose
    
    def fit(self, X) :
        '''
        Applies the fit to dIVA_L
        '''
        self.W, self.iteration, self.cost = diva_l(X)
        
        Y = [[np.zeros(shape=X.shape) for X in X[p]] for p in range(len(X))]
         
        for p in range(len(X)) :
            for k in range(len(X[p])) :
                Y[p][k][:,:] = np.dot(self.W[p][k][:,:], X[p][k][:,:])
         
        return Y

    
    def fit_transform(self, X) :
        
        self.fit(X)
        self.transform(X)

    
    
    
    