
# coding: utf-8

# In[1]:


import randmv_laplace as mvl
import numpy as np

def IVA_L (X, alpha0=0.1, termThreshold=1e-6, termCrit='ChangeInW',
           maxIter=1024, initW=[], A=[], verbose=False, whiten=False) :
    '''
    
    Solves the Independent Vector Analysis with Laplace 
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
            
    X = Data observations from K data sets, i.e. X{k}=A{k}*S{k}
        where A{k} is NxN unknown invertible mixing matrix and
        S{k} is a NxT matrix with the nth row corresponding to
        T samples of nth source in the kth dataset. For IVA, 
        assumed that source is tatistically independent of all 
        sources within dataset and exactly dependent on at 
        most one source in each of the other datasets. The
        data, X, is a 3 dimensional array of dimension NxKxT. 
        The latter enforces the assumption of equal number of 
        samples in each dataset
    
    
    Outputs:
    -------------------------------------------------------------------
    W = Unmixing matrix. 
    
    cost = vector that contains costs associated to each 
        iteration
    
    '''
    
    print termCrit
    
    if (termCrit != 'ChangeInW') and (termCrit != 'ChangeInCost') :
        raise ValueError ('''termCrit has to be either 
                          'ChangeInW' or 'ChangeInCost' ''')
    
    
    ## In the MATLAB script, had N,T,K, but python shape 
    ## ## function returns in (depth, rows, columns) form, not 
    ## ## rows, columns, deep form. Rotate to compensate
    K,N,T = X.shape
    
    W = np.random.rand(K,N,N)
    
    
    ## Initializing variables
    cost = np.array([np.NaN for x in range(maxIter)])
    Y = X * 0.0
    
    ## Main Loop
    for iteration in range(maxIter) :
        termCriterion = 0
        
        ## Initial approximation to true source vectors
        for i in range(1, K) :
            Y[i,:,:] = np.dot(W[i,:,:], X[i,:,:])
        
        
        ## Initializing values for the iteration
        ## Summing over datasets, left with N x T
        ## dataset.
        #######
        ####### Post look through note: Why are we taking abs when we are just doing element-wise multiplication?
        #######
        sqrtYtY = np.sqrt(np.sum(abs(Y)*abs(Y),0))
        sqrtYtYInv = 1 / sqrtYtY
        W_old = W.copy()
        dW    = W*0
        
        ## Computing change in W
        for i in range(K) :
            phi = sqrtYtYInv * Y[i,:,:]
            dW[i,:,:] = W[i,:,:] - np.dot(phi, np.dot(Y[i,:,:].T, W[i,:,:])) / T
        
        ## Updating W
        W = W + alpha0 * dW
        
        ## Computing costs
        cost[iteration] = 0
        for i in range(K) :
            cost[iteration] += np.log(abs(np.linalg.det(W[i,:,:])))
        
        cost[iteration] = np.sum(np.sum(sqrtYtY))/T - cost[iteration]
        cost[iteration] = cost[iteration] / (N*K)
        
        ## Check termination Criterion
        if termCrit == 'ChangeInW' :
            for i in range(K) :
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
        elif np.isnan(cost[iteration]) :
            if verbose :
                print ("W blew up, restarting with new initial value")
            
            for i in range(K) :
                W[i,:,:] = np.identity(N) + 0.1 * rand(N)
            
        elif iteration > 1 and cost[iteration] > cost[iteration-1] :
            alpha0 = max(alphaMin, alphaScale * alpha0)
            
        
        ## Display iteration information
        if verbose :
            print "Step: %i \t W change: %f \t Cost %f" % (iteration, termCriterion, cost(iteration))
        
        ## End iteration
    
    ## Finish display
    if verbose :
        print "Algorithim converged, end results are: "
        print "Step: %i \n W change: %f \n Cost %f \n\n" % (iteration, termCriterion, cost(iteration))
        
    return W




def test_IVA_L ( ) :
    '''
    Takes no inputs, prints sourec vectors, unmxing matrix, what the 
        true unmixing matrix should look like, and what the unmixing 
        matrix that IVA_L returned looks like
    '''
    
    
    print "Running test_IVA_L function"
    
    N=3
    K=10
    T=10000
    
    S=np.zeros(shape=(K,N,T))
    
    for n in range(N) :
        Z = mvl.randmv_laplace(K,T)
        S[:,n,:] = Z
    
    
    A = np.random.rand(K,N,N)
    X = S.copy()
    
    for k in range(K) :
        A[k,:,:] = np.transpose(vecnorm(A[k,:,:]))
        X[k,:,:] = np.dot(A[k,:,:], S[k,:,:])
    
    print "The source vectors S are \n", S, "\n\n"
    print "The mixing matrix A is \n", A, "\n\n"
    
    for k in range(K) :
        print "The true unmixing matrix W for site %i is \n" % k, np.linalg.pinv(A[k,:,:]), "\n"
    
    
    W = IVA_L (X)

    print "IVA_L found that the unmixing matrix W is \n", W


def vecnorm(A) :
    ''' 
    Takes a matrix of vectors and produces a matrix with the same span,
        but with every vector normalized.
        
    Inputs:
    ------------------------------------------------------------------------------
    A = matrix of vectors. A must be a D x N matrix, cannot be multidimensional.
    
    Outputs:
    ------------------------------------------------------------------------------
    A = D x N matrix of vectors. note that every column vector in A has the same 
        span as every corresponding column vector in B.
        
    '''
    
    N = A.shape[1]
    
    
    for n in range(N) :
        A[:,n] = A[:,n] / np.sqrt(np.sum(A[:,n] * A[:,n], axis=0))
    
    return A
    
    
    
if __name__ == "__main__" :
    print test_IVA_L ()

