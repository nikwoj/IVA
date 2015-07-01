import numpy as np
import IVA_L as iva

# def diva_l(X, comp_dict=dict(), alpha0=0.1, max_iter=1024) :
#     '''
#     Implements the distributed IVA_L. 
    
#     Inputs:
#     -------
#     X : List of data matrices, all of same size.
    
#     Outputs:
#     --------
#     W : List of 3-D unmixing matrices for every subject in every site. 
#     '''
    
#     ## If comp_dict is empty, initialize it to be identity, ie
#     ## comp_dict[(comp_num,subj,site)] = comp_num
#     ## Do we also need to assume same number of subjects per site?
#     if len(comp_dict) == 0 :
#         keys = [(n,k,p) for n in range(N) for k in range(K) for p in range(P)]
#         comp_dict = {k: 1 for k in keys}
    
#     alpha_min   = 0.1
#     alpha_scale = 0.9
    
#     ## IVA assumes that every subject is same size
#     N,T,K = X[0].shape
#     P = len(X)
    
#     W = []
#     Y = []
    
#     for p in range(P) :
#         W.append(np.random.rand(N,N,K))
#         Y.append(X[p].copy())
    
def diva_l (X, verbose=False, A=[]) :
    '''
    Implements the distributed IVA_L. 
    
    Inputs:
    -------
    X : List of data matrices, all of same size.
    
    Outputs:
    --------
    W : List of 3-D unmixing matrices for every subject in every site. 
    '''
    N,T,K,P    = X.shape
    Y          = np.zeros(shape=(N,T,K,P))
    Y_per_site = np.zeros(shape=(N,T,P))
    
    if A != [] :
        supply_A = True
        
        if A.shape != (N,N,K,P) :
            raise ValueError ("A needs to be right shape")
    
    unmixing_site = np.zeros(shape=(N,N,K,P))
    
    guess = np.zeros(shape=(N,N,P))
    
    ## Do site specific IVA, and initalize variables
    for p in range(P) :
        
        if verbose :
            print (" Running IVA for site %i" % p)
        
        ## Create and fit model
        W, iterations, _= iva.iva_l(X[:,:,:,p])
        
        
        for k in range(K) :
            Y[:,:,k,p] = np.dot(W[:,:,k], X[:,:,k,p])
        
        if verbose :
            print iterations
        
        if supply_A :
            for k in range(K) :
                print np.dot(W[:,:,k], A[:,:,k,p])
        
        ## Save W matrix for specific site
        unmixing_site[:,:,:,p] = W
        
        ## Prep the subject to send by averaging all subjects
        Y_per_site[:,:,p] = np.sum(Y[:,:,:,p],2) / K`
        
        ## While we're at it, create Initial W matrix
        guess[:,:,p] = np.identity(N)
    
    
    ## Remove uneeded variables, or will python garbage collection do that 
    ## for me? Could just manually rm X, specifically
    
    if verbose :
        print ("Starting final master node IVA")
    
    W, iterations, _ = iva.iva_l(Y_per_site, W_init = guess)
    
    if verbose :
        print ("Master node converged in %i iterations" % iterations)
    
    for p in range(P) :
        for k in range(K) :
            unmixing_site[:,:,k,p] = np.dot(W[:,:,p], unmixing_site[:,:,k,p])
    
    return unmixing_site
    
    
    
    # true_sources = []
    
    # S = np.zeros(shape=(N,T,K))
    
    # for p in range(P) :
    #     for k in range(K) :
    #         S[:,:,k] = np.dot(final_IVA.unmixing()[p], tot_Y_site[p][:,:,k])
    #     true_sources.append(S)
    
    
    