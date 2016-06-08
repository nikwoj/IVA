## Just general need modules
import numpy as np
from numpy import dot, zeros
from ica import pca_whiten
## for testing purposes, data was generated in MATLAB
## Will create (or rather, fix...) generating function at later date


def _get_dW (W, Y, sqrtYtYInv) :
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
    ## T is number of samples    
    _, T, K = Y.shape
    dW = W.copy()
    
    for k in range(K) :
        phi = sqrtYtYInv * Y[:,:,k]
        dW[:,:,k] = W[:,:,k] - np.dot( np.dot(phi, np.transpose(Y[:,:,k]) / T), W[:,:,k])
    
    return dW


def _vecnorm (A) : return normalize(A, axis=0, norm='l2')
''' Note: Only used when test_IVA_L is called in order to orthonomalize mixing matrix '''


def _compute_cost (W, sqrtYtY) :
    '''
    Outputs the cost of the current iteration
    
    Inputs:
    -------
    W: The unmixing matrix
    
    sqrtYtYInv: Summary of the current Y across sites
    
    Outputs:
    --------
    current_cost: A number associated to the cost of the current iteration.
    '''
    
    N, N, K = W.shape
    T = sqrtYtY.shape[1]
    current_cost = 0
    
    for k in range(K) :
        current_cost = current_cost + np.log(abs(np.linalg.det(W[:,:,k])))
    
    current_cost = (-1)*current_cost + np.sum(np.sum(sqrtYtY,1)) / T
    current_cost = current_cost / (N*K)
    
    return current_cost



def iva_l (X, W_init, term_threshold=1e-6, term_crit='ChangeInCost',
           max_iter=2048, A = [], verbose=False, n_components=0) :
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
        N,T,K = X.shape
    except ValueError :
        raise ValueError ('''X needs to be 3-D, or in (N,T,K) form. 
                             current matrix is %s''' % str(X.shape))
    
    cost = [np.NaN for x in range(max_iter)]
    if n_components > 0 :
        wht = zeros((n_components, N, K))
        de_wht = zeros((N, n_components, K))
        X_white = zeros((n_components, T, K))
        for k in range(K) :
            X_white[:,:,k], wht[:,:,k], de_wht[:,:,k] = pca_whiten(X[:,:,k], n_components, verbose=verbose)
        X = X_white
    Y = X.copy()
    N,T,K = X.shape
    
    if W_init == [] :
        W = np.random.rand(N,N,K)
    else :
        if W_init.shape == (N,N,K) :  
            W = W_init
        else :
            raise ValueError ('''W has to have dimension %i x %i for
                    each of the %i sites, in form (rows, columns. subjects)
                    \n W defaulting to random.''' % (N, N, K))
    
    if (term_crit != 'ChangeInW') and (term_crit != 'ChangeInCost') :
        raise ValueError ('''term_crit has to be either 
                          'ChangeInW' or 'ChangeInCost' ''')
    
    alphamin   = 0.01
    alpha_scale = 0.9
    alpha0     = 0.1
    
    back_num = 10.0
    backtrack = False
    ## Main Loop
    for iteration in range(max_iter) :
        term_criterion = 0
        
        ## Initial approximation to true source vectors
        for k in range(K) :
            Y[:,:,k] = np.dot(W[:,:,k], X[:,:,k])
            # Y[:,:,k] += np.random.laplace(size = (N,T)) *2
        
        ## Initializing values for the iteration summing over datasets, left with N x T
        ## dataset.
        sqrtYtY    = np.sqrt(np.sum(Y*Y,2))
        
        sqrtYtYInv = 1 / sqrtYtY
        
        cost[iteration] = _compute_cost(W, sqrtYtY)
        
        if iteration > 1 :                        
            if backtrack == True :
                if cost[iteration] < min(cost[0:iteration]) :
                    backtrack = False
            else :
                if cost[iteration] > min(cost[0:iteration]) :
                    backtrack = True
        
        if backtrack == False :
            dW = (1/back_num) * _get_dW(W, Y, sqrtYtYInv)
        else :
            W  -= dW
            dW *= 1.0/2.0
            back_num += 1
        
        W += dW
        
        ## Check termination Criterion
        if term_crit == 'ChangeInW' :
            for k in range(K) :
                tmp_W = W_old[:,:,k] - W[:,:,k]
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
        
        ## Display iteration information
        if verbose :
            print "Step: %i \t W change: %f \t Cost %f" % (iteration, term_criterion, cost[iteration])
            
            ## End iteration
    
    
    return sqrtYtYInv, cost
    
    
    if iteration==max_iter :
        print ('''Algorithm may have not converged, reached max
               number of iterations ''')
    print "a"
    ## Finish display
    if verbose :
        print "Algorithim converged, end results are: "
        print " Step: %i \n W change: %f \n Cost %f \n\n" % (iteration, term_criterion, cost[iteration])
    if n_components > 0 :
        return W, wht, [de_wht, cost]
    else :
        return W


    
class IVA_L( ) :
    '''
    IVA_L is the Independent Vector Analysis using multivariate Laplacian 
        distribution
    
    Inputs:
    -------
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
    
    Methods:
    --------
    fit : 
    
    transform :
    
    fit_transform :
    
    unmixing :
    
    mixing :
    
    iteration :
    
    Attributes:
    -----------
    unmixing : Returns the unmixing matrix for the data. 
    
    mixing : Returns the mixing matrix for the transformed data, ie inverse of unmixing matrix
        
    iteration : Number of iterations taken to converge. 
    '''
    
    def __init__(self, alpha0=0.1, term_threshold=1e-6, term_crit='ChangeInCost',
                 max_iter=1024, W_init=[], verbose=False) :
        
        self.alpha0         = alpha0
        self.term_threshold = term_threshold
        self.term_crit      = term_crit
        self.max_iter       = max_iter
        self.W_init         = W_init
        self.verbose        = verbose
        
    def fit(self, X) :
        '''
        Finds the W matrices to transform the data X
        
        Inputs:
        -------
        X: Data that is to be transformed. Should be in 
            (rows, columns, depth) form.
        '''
        
        self.W, self.iteration, self.cost = iva_l(
            X, self.alpha0, self.term_threshold, self.term_crit, 
            self.max_iter, self.W_init, self.verbose
        )
    
    
    def transform(self, X) :
        '''
        Takes in data akin to data used to fit W, and transforms data into source vectors
        
        Inputs:
        -------
        X : array, shape=(N,T,K)
            Data matrix. 
            
        Outputs:
        --------
        Y : array, shape=(N,T,K)
            Transformed data matrix.
        '''
        Y = X.copy()
        
        try :
            K = X.shape[2]
            for k in range(K) :
                Y[:,:,k] = np.dot(self.W[:,:,k], X[:,:,k])
            return Y
        
        except AttributeError :
            self.fit_error("W")
    
    
    def fit_transform(self, X) :
        '''
        Takes in data, fits unmixing matrix W, then transforms data.
        
        Inputs:
        -------
        X : array, shape=(N,T,K)
            Data matrix which IVA_L will be applied to
            
        Outputs:
        --------
        Y : array, shape=(N,T,K)
            Data matrix of true sources. 
        '''
        
        self.fit(X)
        K = X.shape[2]
        Y = X.copy()
        for k in range(K) :
            Y[:,:,k] = np.dot(self.W[:,:,k], X[:,:,k])
        return Y
        
    def unmixing(self) :
        '''
        Returns the current unmixing matrix W. 
        
        Inputs:
        -------
        Nothing
        
        Outputs:
        --------
        W : array, shape = (N,N,K)
            The unmixing matrix
        '''
        
        try :
            return self.W
        except AttributeError :
            self.fit_error("W")
            
            
    def mixing(self) :
        '''
        Returns the pseudo inverse of the unmixing matrix W.
        
        Inputs:
        -------
        Nothing
        
        Outputs:
        --------
        A : array, shape = (N,N,K)
            The mixing matrix
        '''
        
        try :
            N,N,K = self.W.shape
            A = W.copy()
            for k in range(K) :
                A[:,:,k] = np.linalg.pinv(W[:,:,k])
            return A
            
        except AttributeError :
            self.fit_error("W")
            
    def cost(self) :
        '''
        Returns the cost associated to the fitting process
        '''
        
    def iteration(self) :
        '''
        Returns the number of iterations it took to converge to solution.
        '''
        
        try :
            return self.iterations
        except AttributeError :
            self.fit_error("iterations")
            
    def fit_error(self, reason) :
        '''
        Returns the error message associated to their problem.
        '''
        
        raise NotImplementedError ("%s not defined. Run fit method before attempting to transform data" % reason)
