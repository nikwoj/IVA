## Just general need modules
import autograd.numpy as np



def set_para(X, verbose) :
    N,T,K = X.shape
    
    def _compute_cost (W) :
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
        if verbose :
            print "Running cost function"
            
        for k in range(K) :
            Y[:,:,k] = np.dot(W[:,:,k], X[:,:,k])
        
        sqrt_YtY = np.sqrt( np.sum(Y*Y, 2))
        
        current_cost = 0
        
        for k in range(K) :
            current_cost = current_cost + np.log(abs(np.linalg.det(W[:,:,k])))
        
        current_cost = (-1)*current_cost + np.sum(sqrtYtY,1) / T
        current_cost = current_cost / (N*K)
        
        return current_cost
    



def iva_l (X, W_init=[], verbose=False) :
    '''
    IVA_L is the Independent Vector Analysis using multivariate Laplacian 
        distribution
    
    Inputs:
    -------
    X : 3-D data matrix. Note that HAS to be 3-D, even if only have one 
        subject.
    
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
    
    ## Possible optimazition area: X.copy() or X * 0 or np.zeros(shape=(N,T,K))
    Y = X.copy()
    alpha_min   = 0.1
    alpha_scale = 0.9
    
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
    
    ## Main Loop
    
    W, _, d = fmin_l_bfgs_b
    
    ## Finish display
    if verbose :
        print "Algorithim converged, end results are: "
        print " Step: %i \n W change: %f \n Cost %f \n\n" % (iteration, term_criterion, cost[iteration])
    
    return W, iteration, cost


    
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