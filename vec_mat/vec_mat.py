import autograd.numpy as np
import stack as st



def mat_to_vec (W, K) :
    '''
    Takes in matrix that needs to be vectorized and does that
    
    Inputs:
    -------
    W : 1-D array
        Vector to be converted into a matrix
    
    K : Integer
        Number of subjects, ie number of W matrices there are
    '''
    tmp_W = []
    for k in range(K) :
        tmp_W.extend(st.stack(W[:,:,k]).tolist())
    
    return np.array(tmp_W)
    


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
        new_W[:,:,k] = st.unstack(np.array(W_subj))
    
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
        new_W.append( st.unstack(W_subj) )
    
    return new_W
