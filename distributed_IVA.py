import numpy as np

def d_IVA_L (X, alpha0=0.1, termThreshold=1e-6, termCrit="ChangeinW",
           maxIter=1024, initW=[], A=[], verbose=False, whiten=False) :
    '''
    
    Computes the Independent Vector Analysis with Laplace 
    distribution given datasets. 
    
    -------------------------------------------------------------------
    -------------------------------------------------------------------
    NOTE: whitten, A, initW currently are not operational
    -------------------------------------------------------------------
    -------------------------------------------------------------------
    
    
    Inputs:
    -------------------------------------------------------------------
    
    Params = A dictionary containing the optional arguments,
        whitten = whitten the data
            (Boolean, True)
        verbose = enable print statements
            (Boolean, False)
        A = True Mixing matrix, auto sets verbose to True 
            (Matrix, [])
        initW = Initial estimates for W matrix
            (Matrix, [])
        maxIter = Maximum number of iterations
            (Integer, 1024)
        termCrit = criteria for terminating loop,
            either "ChangeInW" or "ChangeInCost".
            (String, 'ChangeInW')
        termThreshold = Termination threshold
            (Float, 1e-6)
        alpha0 = Initial step size scaling
            (Float, 0.1)
            
    X = Data observations from K data sets across L sites. The site 
        should be the fourth dimension, each individual dataset per
        site should be in the third dimension. 
        
        NOTE: Due to numpy stupitidty, a four dimensional array with 
        shape=(2,3,4,5) will have two matrices with each matrix having
        three matrices with four rows and five columns. I know it 
        doesn't make sense, Don't ask questions, I just program here.
    
    
    
    Outputs:
    -------------------------------------------------------------------
    W = Unmixing matrices. 
    
    cost = vector that contains costs associated to each 
        iteration
    
    '''
    
    if (terminationCriterion != 'ChangeInW') and (terminationCriterion 
                                                  != 'ChangeInCost') :
        raise ValueError ('''terminationCriterion has to be either 
                          'ChangeInW' or 'ChangeInCost' ''')
    
    
    
    ## In the MATLAB script, had N,T,K for rows, columns, depth, 
    ## but python shape function returns in (depth, rows, column)
    ## form, not rows, columns, deep form. Rotate to compensate, 
    ## add extra dimension for sites
    L,K,N,T = X.shape
    

    ## Inititializing variables. W is site specific unmixing matrices.
    ## Y is site specific approximation to independent sources
    ## Y_hat is site specific approximations sent to global node
    ## W_glb is global unmixing matrix
    ## dW is matrix that will be multiplied with W to produce new 
    ## ## approximation to W, iteratively
    ## S is independent sources for global IVA
    W     = np.random.rand(shape=(L,K,N,N))
    Y     = X * 0.0
    Y_hat = X[:, 1, :, :] * 0.0
    W_glb = np.zeros(shape=(L,N,N))
    dW    = np.zeros(shape=(K,N,N))
    S     = np.zeros(shape=(L,N,T))
    
    
    for i in range(L) :
      for j in range(K) :
          W_glb[i,j,:,:] = np.identity(N)

    
        
    ## Vector that has costs of every computation
    cost = np.array([np.NaN for x in range(maxIter)])
    
    
    ## Permute array that tells which components to take from which 
    ## subjects. Refreshed every iteration, and for every site
    ##
    ## Shape refers to number of sites and number of components
    permute = np.zeros(shape=(L,1,N))
    
    
    ## Main Loop
    for iteration in range(maxIter) :
        termCriterion = 0

        ## Load the permute array for the current iteration
        for l in range(L) :
            permute[l,:,:] = np.array([np.random.permutation(N) % K])
        
        ## Initial approximation to true source vectors
        for l in range(L) :
            for k in range(K) :
                Y[l,k,:,:] = np.dot(W_glb[l,k,:,:], np.dot(W[l,k,:,:], X[l,k,:,:]))
                
            
            ## Basically, says that for site l we want the nth component from 
            ## permute[n] subject to be sent to global.
            for n in range(N) :
                Y_hat[l,n,:] = Y[l,permute[n],n,:]
            
            
            ## In future implementation, may base updated W on what happens after
            ## computing site specific IVA. In that case, delete this part
            sqrtYtY    = np.sqrt(np.sum(Y[l,:,:,:]*Y[l,:,:,:],0))
            sqrtYtYinv = 1 / sqrtYtY
            
            ## W_old is computed for cost computations. If don't care about that,
            ## then can just delete it.
            W_old = W.copy()
            
            ## Computing change in W
            for k in range(K) :
                phi = sqrtYtYInv * Y[l,k,:,:]
                dW[l,k,:,:] = W[l,k,:,:] - np.dot(phi, np.dot(Y[l,k,:,:].T), W[l,k,:,:]) / T
            
            ## Updating W
            W[l,:,:,:] = W[l,:,:,:] + alpha0 * dW[l,:,:,:]
        
        
        ## This is where we are getting initial approximations
        ## for the sites. Send these to master node, have it
        ## compute its own approximations on Global level.
        
        for l in range(L) :
            S[l,:,:] = W_glb[l,:,:] * Y_hat[l,:,:]
            
        sqrtStS    = np.sqrt(np.sum(S * S, axis=0))
        sqrtStSinv = 1 / sqrtStS
        
        W_glb_old = W_glb.copy()
        dW_glb    = W_glb * 0
        
        for l in range(L) :
            phi_glb = sqrtStSInv * S[l,:,:]
            dW_glb[l,:,:] = W_glb[l,:,:] - np.dot(phi, np.dot(Y[l,:,:].T), W_glb[l,:,:]) / T
            
        W_glb = W_glb + alpha0 * dW_glb
        
        ## Do I worry about a cost analysis? Seems VERY intensive to write....
        ## Not worrying about it for now.
        #####
        #####
        ##### NOTE: There is currently no cost analysis happening, so if code has huge cost computation, will go unnoticed.
        #####
        #####






    return W

