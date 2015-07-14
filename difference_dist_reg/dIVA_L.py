## Just general need modules
import numpy as np
from sklearn.preprocessing import normalize

## for testing purposes, data was generated in MATLAB
## Will create (or rather, fix...) generating function at later date
import scipy.io as sio


def _vecnorm(A) : return normalize(A, axis=0, norm='l2')
''' Note: Only used when test_IVA_L is called in order to orthonomalize mixing matrix '''


def _get_dW (W, Y, sqrtYtYInv) :
    '''
    Computes the change matrix dW. 
    
    A local function, doesn't need to be computed at Master Node.
    
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
    N, T, K, P = Y.shape
    dW = np.zeros(shape=(N,N,K,P))
    
    for p in range(P) :
        for k in range(K) :
            phi = np.dot(sqrtYtYInv * Y[:,:,k,p], np.transpose(Y[:,:,k,p])) / T
            dW[:,:,k,p] = W[:,:,k,p] - np.dot(phi, W[:,:,k,p])
    
    return dW


def _compute_cost(W, sqrtYtY) :
    '''
    Outputs the cost of the current iteration
    
    A local function, doesn't need to be computed at Master Node
    
    Inputs:
    -------
    W: The unmixing matrix
    
    sqrtYtYInv: Summary of the current Y across sites
    
    Outputs:
    --------
    current_cost: A number associated to the cost of the current iteration.
    '''
    
    ## Is dummy assignment faster than just single assignemtn?
    N, N, K, P = W.shape
    T = sqrtYtY.shape[1]
    current_cost = 0
    
    for p in range(P) :
        for k in range(K) :
            current_cost = current_cost + np.log(abs(np.linalg.det(W[:,:,k,p])))
    
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
    X : 3-D data matrix. Note that HAS to be 3-D, even if only have one 
        subject.
    
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
    W : array, shape = (N,N,K)
        The unmixing matrix W
    '''
    
    try :
        N,T,K,P = X.shape
    except ValueError :
        raise ValueError ('''X needs to be 4-D, or in (N,T,K,P) form. 
                             current matrix is %s''' % str(X.shape))
    
    cost = [np.NaN for x in range(max_iter)]
    
    ## Make Y have same dimension as X. Equivalent to X.copy(), but faster    
    Y = np.zeros(shape=(N,T,K,P))
    alpha_min   = 0.1
    alpha_scale = 0.9
    
    if W_init == [] :
        W = np.random.rand(N,N,K,P)
    else :
        if W_init.shape == (N,N,K,P) :  
            W = W_init
        else :
            raise ValueError ('''W has to have dimension %i x %i for
                    each of the %i sites, in form (rows, columns. subjects)
                    \n W defaulting to random.''' % (N, N, K, P))
    
    if (term_crit != 'ChangeInW') and (term_crit != 'ChangeInCost') :
        raise ValueError ('''term_crit has to be either 
                          'ChangeInW' or 'ChangeInCost' ''')
    
    ## Main Loop
    for iteration in range(max_iter) :
        term_criterion = 0
        
        YtY = np.zeros(shape=(N,T,P))
        
        ## Happens Locally
        for p in range(P) :
            for k in range(K) :
                Y[:,:,k,p] = np.dot(W[:,:,k,p], X[:,:,k,p])
            
            YtY[:,:,p] = np.sum(Y[:,:,:,p]*Y[:,:,:,p],2)
        
        ## Happens at Master Node
        sqrtYtY    = np.sqrt(np.sum(YtY, 2))
        sqrtYtYInv = 1 / sqrtYtY
        
        W = W + alpha0 * _get_dW(W, Y, sqrtYtYInv)
        
        cost[iteration] = _compute_cost(W, sqrtYtY)
        
        ## Check termination Criterion
        if term_crit == 'ChangeInW' :
            for p in range(P) :
                for k in range(K) :
                    tmp_W = W_old[:,:,k,p] - W[:,:,k,p]
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
            for k in range(K) :
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
        self.W, self.iteration, self.cost = diva_l(X)
    
    def transform(self, X) :
        _, _, K, P = X.shape
        try :
            for p in range(P) :
                for k in range(K) :
                    X[:,:,k,p] = np.dot(self.W[:,:,k,p], X[:,:,k,p])
            
            return X
        except AttributeError : #Include what the error should actually be?
            self._error("fit")
    
    def fit_transform(self, X) :
        self.W, self.iteration, self.cost = diva_l(X)
        
    
    def fit_error(self, reason) :
        raise 
    
    '''
    
    To do: find a correct method of 
    
    
    
    
    '''