try :
    import autograd.numpy as np
except :
    import numpy as np

def stack (W) : 
    ## Takes matrix and outputs vector
    return W.reshape(W.shape[0]**2)

def unstack(W) :
    ## Takes vector and outputs matrix stacked according to stack
    W = np.array(W)
    N = np.sqrt(W.shape)
    return W.reshape(N,N)

def mat_to_vec (W) :
    '''
    Takes in matrix that needs to be vectorized and does that
    
    Inputs:
    -------
    W : 1-D array
        Vector to be converted into a matrix
    
    K : Integer
        Number of subjects, ie number of W matrices there are
    '''
    try :
        tmp_W = []
        K = W.shape[2]
        for k in range(K) :
            tmp_W.extend(stack(W[:,:,k]).tolist())
        
        return np.array(tmp_W)
    except :
        P = len(W)
        K = [w.shape[2] for w in W]
        tmp_W = []
        for p in range(P) :
            for k in range(K[p]) :
                tmp_W.extend(stack(W[p][:,:,k]).tolist())
        return tmp_W

def vec_to_mat (W, N, K) :
    '''
    Takes in a vector that needs to be converted to a matrix and 
        does that.
    
    Inputs:
    -------
    W : 1-D array
        Vector to be converted into a matrix
    
    N : Integer
        Number of rows that each W matrix needs to have
    
    K : Integer
        Number of subjects, ie number of W matrices there are
    '''
    new_W = np.zeros((N,N,K))
    for k in range(K) :
        W_subj = W[k*N*N:(k+1)*N*N]
        new_W[:,:,k] = unstack(np.array(W_subj))
    
    return new_W

def vec_to_mat_l (W, N, K) :
    '''
    Takes in a vector that needs to be converted to a list of
        2-D matrices and does that.
    
    Inputs:
    -------
    W : 1-D array
        Vector to be converted into a matrix
    
    N : Integer
        Number of rows that each W matrix needs to have
    
    K : Integer
        Number of subjects, ie number of W matrices there are
    '''
    new_W = []
    for k in range(K) :
        W_subj = W[k*N*N:(k+1)*N*N]
        new_W.append( unstack(W_subj) )
    
    return new_W
