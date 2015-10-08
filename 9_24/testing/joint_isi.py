import numpy as np


def joint_ISI (W,A) :
    '''
    Takes in unmixing and mixing matrix and computes the
        joint ISI, or measure of independence (I believe)
    
    Inputs:
    -------
    W : 3-D or 4-D array or list of lists of 2-D arrays
        The unmixing matrices, can be contained in a 3-D
        array as in the case of normal IVA, a 4-D array
        as in the case of version3 distributed IVA, or
        a list of lists of 2-D arrays as in the case of 
        version4 distributed IVA
    '''
    
    ## If W is just a single site, then
    try :
        N,N,K = W.shape
        B = np.zeros((N,N))
        
        for k in range(K) :
            B[:,:] += np.abs(np.dot(W[:,:,k], A[:,:,k]))
        
        row_sum = 0
        col_sum = 0
        
        for n in range(N) :
            row_max = np.max(B[n,:])
            col_max = np.max(B[:,n])
            
            row_sum += np.sum(B[n,:] / row_max) - 1
            col_sum += np.sum(B[:,n] / col_max) - 1
        
        tot_sum = (row_sum + col_sum) / (2 * N * (N-1))
        return tot_sum

    ## If instead W is multiple sites, then
    except ValueError :
        N,N,K,P = W.shape

        for k in range(K) :
            for p in range(P) :
                W[:,:,k,p] = np.abs(np.dot(W[:,:,k,p], A[:,:,k,p]))
        
        W = np.sum(np.sum(W,3),2)
        
        row_sum = 0
        col_sum = 0
        
        for n in range(N) :
            row_max = np.max(W[n,:])
            col_max = np.max(W[:,n])
            
            row_sum += np.sum(W[n,:] / row_max) - 1
            col_sum += np.sum(W[:,n] / col_max) - 1
        
        tot_sum = (row_sum + col_sum) / (2 * N * (N-1))
        return tot_sum
    
    except AttributeError :
        P = len(W)
        K = [W[p].shape[2] for p in range(P)]
        N, N, _ = W[0][:,:,:].shape
        B = np.zeros(shape=(N,N))
        
        
        for p in range(P) :
            for k in range(K[p]) :
                B += np.abs(np.dot(W[p][:,:,k], A[p][:,:,k]))
        
        row_sum = 0
        col_sum = 0
       
        for n in range(N) :
            row_max = np.max(B[n,:])
            col_max = np.max(B[:,n])
           
            row_sum += np.sum(B[n,:] / row_max) - 1
            col_sum += np.sum(B[:,n] / col_max) - 1
        
        tot_sum = (row_sum + col_sum) / (2 * N * (N-1))
        return tot_sum
