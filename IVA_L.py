
# coding: utf-8

# In[1]:

import scipy.io as sio
import randmv_laplace as mvl
import numpy as np
a = sio.loadmat("variables.mat")


# In[2]:

def IVA_L (X, alpha0=0.1, termThreshold=1e-6, termCrit='ChangeInCost',
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
    
    alphaMin   = 0.1
    alphaScale = 0.9
    if (termCrit != 'ChangeInW') and (termCrit != 'ChangeInCost') :
        raise ValueError ('''termCrit has to be either 
                          'ChangeInW' or 'ChangeInCost' ''')
    
    
    ## In the MATLAB script, had N,T,K, but python shape 
    ## ## function returns in (depth, rows, columns) form, not 
    ## ## rows, columns, deep form. Rotate to compensate
    K,N,T = X.shape

#     W = np.random.rand(K,N,N)
    W = np.zeros(shape=(K,N,N))
    for k in range(K) :
        W[k,:,:] = np.identity(N) 
    
    ## Initializing variables
    cost = np.array([np.NaN for x in range(maxIter)])
    Y = X * 0.0
    
    ## Main Loop
#     for iteration in range(maxIter) :
    for iteration in range(maxIter) :
        termCriterion = 0
        
        ## Initial approximation to true source vectors
        for i in range(K) :
            Y[i,:,:] = np.dot(W[i,:,:], X[i,:,:])
        
        
        ## Initializing values for the iteration summing over datasets, left with N x T
        ## dataset.
        sqrtYtY = np.sqrt(np.sum(Y*Y,0))
        sqrtYtYInv = 1 / sqrtYtY
        
        ## Save current W as W_old
        W_old = W.copy()
        dW    = W*0
        
        ## Computing change in W
        for k in range(K) :
            phi = sqrtYtYInv * Y[k,:,:]
            dW[k,:,:] = W[k,:,:] - np.dot( np.dot(phi, np.transpose(Y[k,:,:]) / T), W[k,:,:])
        
        ## Updating W
        W = W + alpha0 * dW
        
        ## Computing costs
        cost[iteration] = 0
        for k in range(K) :
            cost[iteration] = cost[iteration] + np.log(abs(np.linalg.det(W[k,:,:])))
        
        cost[iteration] = np.sum(np.sum(sqrtYtY)) / T - cost[iteration]
        cost[iteration] = cost[iteration] / (N*K)
        
        ## Check termination Criterion
        if termCrit == 'ChangeInW' :
            for k in range(K) :
                tmp_W = W_old[k,:,:] - W[k,:,:]
                termCriterion = max(termCriterion, np.linalg.norm(tmp_W[:,:]))
                
                ## Old criterion, updated.
#                 termCriterion = max(termCriterion, max(1-np.abs(np.diag(np.dot(W_old[k,:,:], W[k,:,:].T)))))
                
        elif termCrit == 'ChangeInCost' :
            if iteration == 1 :
                termCriterion = 1
                
            else :
                termCriterion = (abs(cost[iteration-1]-cost[iteration])
                                 / abs(cost[iteration]))
        
        
        ## Check termination condition
        if termCriterion < termThreshold or iteration == maxIter :
            break
        elif np.isinf(cost[iteration]) :
            if verbose :
                print "W blew up, restarting with new initial value"
                print "For reference, Here is W."
                print W
                
            
            for i in range(K) :
                W[i,:,:] = np.identity(N) + 0.1 * np.random.rand(N)
            
        elif iteration > 1 and cost[iteration] > cost[iteration-1] :
            alpha0 = max(alphaMin, alphaScale * alpha0)
            print "Changing the alpha0"
            print alpha0
        
        ## Display iteration information
        if verbose :
            print "Step: %i \t W change: %f \t Cost %f" % (iteration, termCriterion, cost[iteration])
        
        ## End iteration
    
    ## Finish display
    if verbose :
        print "Algorithim converged, end results are: "
        print " Step: %i \n W change: %f \n Cost %f \n\n" % (iteration, termCriterion, cost[iteration])
        
    return W


# In[3]:

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


# In[4]:

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
    
    ## We are currently, in an attempt to compare matlab and python code
    ## for errors, controlling S and A, but otherwise, this is the right 
    ## code.
#     S = np.zeros(shape=(K,N,T))
    
#     for n in range(N) :
#         Z = mvl.randmv_laplace(K,T)
#         S[:,n,:] = Z
        
#     A = np.random.rand(K,N,N)
#     X = s.copy()
    
    A = np.zeros(shape=(10,3,3))

    A = a['A']
    S = a['S']
    A_py = np.zeros(shape=(10,3,3))
    S_py = np.zeros(shape=(10,3,10000))
    
    for i in range(3) :
        for k in range(10) :
            for j in range(3) :
                A_py[k,i,j] = A[i,j,k]
            for j in range(10000) :
                S_py[k,i,j] = S[i,j,k]

    X = S_py.copy()
    
    for k in range(K) :
#         A[k,:,:] = np.transpose(vecnorm(A[k,:,:]))
        X[k,:,:] = np.dot(A_py[k,:,:], S_py[k,:,:])
        
#     print "The source vectors S are \n", S, "\n\n"
#     print "The mixing matrix A is \n", A, "\n\n"
    
#     for k in range(K) :
#         print "The true unmixing matrix W for site %i is \n" % k, np.linalg.pinv(A_py[k,:,:]), "\n"
    
    
    W = IVA_L (X, verbose=True)

    print "IVA_L found that the unmixing matrix W is \n", W, "\n\n"
    
    for k in range(K) :
        print "For site %i, the unmixing matrix W times A is \n" % k

        print np.dot(W[k,:,:], A_py[k,:,:]), "\n"
        
#     for k in range(K) :
#         print "For site %i, IVA_L found that the true source vectors are \n" % k
        
#         print np.dot(W[k,:,:], X[k,:,:])


# In[5]:

if __name__ == "__main__" :
    print test_IVA_L ()

