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
    W = Unmixing matrix. 
    
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
    
    
    ## Might be more effiecent to just have them be ndarrays, but
    ## I'll try dictionaries first, as they are easier to read
    # W     = dict()
    # Y     = dict()
    # Y_hat = dict()
    # W_glb = dict()
    
    
    ## This is the other way it could be done, without using
    ## dictionaries
    W     = np.random.rand(shape=(L,K,N,N))
    Y     = X * 0.0
    Y_hat = X[:, 1, :, :] * 0.0
    W_glb = np.zeros(shape=(L,K,N,N))

    for i in range(L) :
      for j in range(K) :
          W_glb[i,j,:,:] = np.identity(N)
    
    
    
    ## For each site, create unmixing matrix in W dataframe, 
    ## independent source matrix for each site in Y dataframe, 
    ## and a new subject matrix in Y_hat dataframe
    # for i in range(L) :
    #     W    ["Site_%i" % i] = np.random.rand(K,N,N)
    #     Y    ["Site_%i" % i] = X[0, :, :, :] * 0.0
    #     Y_hat["Site_%i" % i] = X[0, 0, :, :] * 0.0
        
        ## Now need to set up all of the Global W matrices
        # for j in range(K) :
        #     W_glb[i, j, :, :] = np.identity(N)
    
    ## Vector that has costs of every computation
    cost = np.array([np.NaN for x in range(maxIter)])
    
    
    ## Main Loop
    for iteration in range(maxIter) :
        termCriterion = 0

        ## Vector that says which components to take from where. 
        ## In perfect world, would have 1 for each site, but for
        ## now this will do. Modulo K because there are at K
        ## datasets. The index determines the component, the value
        ## at the component determines which dataset to take the 
        ## result from.
        permute = np.array([np.random.permutation(N) % K for x in range(K)])
        
        ## Initial approximation to true source vectors
        for i in range(L) :
            for j in range(K) :
                Y[i,j,:,:] = np.dot(W[i,j,:,:], X[i,j,:,:])
                
        
        
        ## This is where we are getting initial approximations
        ## for the sites. Send these to master node, have it
        ## compute its own approximations on Global level.
        
        




        
#########################################
#########################################
#########################################
#########################################
#########################################
#########################################
#########################################
#########################################
#########################################
#########################################
#########################################





        ## Initializing values for the iteration
        sqrtYtY = np.sqrt(np.sum(abs(Y)*abs(Y),0))
        sqrtYtYinv = 1 / sqrtYtY
        W_old = W.copy()
        dW    = W*0
        
        ## Computing change in W
        for i in range(K) :
            phi = sqrtYtYInv * Y[i,:,:]
            dW[i,:,:] = W[i,:,:] - np.dot(phi, np.dot(Y[i,:,:].T), W[i,:,:]) / T
        
        
        
        ## Updating W
        W = W + alpha0 * dW
        
        ## Computing costs
        cost[iteration] = 0
        for i in range(k) :
            cost[iteration] += np.log(abs(det(W[i,:,:])))
        
        cost[iteration] = np.sum(np.sum(sqrtYtY))/T - cost[iteration]
        cost[iteration] = cost[iteration] / (N*K)
        
        ## Check termination Criterion
        if termCrit == 'ChangeInW' :
            for i in range(k) :
                termCriterion = max(termCriterion, max(1-np.abs(np.diag(np.dot(W_old[i,:,:], W[i,:,:].T)))))
                
        elif termCrit == 'ChangeInCost' :
            if iteration == 1 :
                termCriterion = 1
                
            else :
                termCriterion = (abs(cost(iteration-1)-cost(iteration))
                                 / abs(cost(iteration)))
        
        
        ## Check termination condition
        if termCriterion < termThreshold or iteration == maxIter :
            break
        elif np.isnan(cost(iteration)) :
            if verbose :
                print ("W blew up, restarting with new initial value")
            
            for i in range(K) :
                W[i,:,:] = np.identity(N) + 0.1 * rand(N)
            
        elif iteration > 1 and cost(iteration) > cost(iteration-1) :
            alpha0 = max(alphaMin, alphaScale * alpha0)
            
        
        ## Display iteration information
        if verbose :
            print "Step: %i \t W change: %f \t Cost %f" % (iteration, termCriterion, cost(iteration))
        
        ## End iteration
    
    ## Finish display
    if verbose :
        print "Algorithim converged, end results are: "
        print "Step: %i \n W change: %f \n Cost %f \n\n" % (iteration, termCriterion, cost(iteration))
        
    return W, cost

