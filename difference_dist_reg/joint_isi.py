import numpy as np


def joint_ISI (W,A) :
    '''
    Takes in unmixing and mixing matrix and computes the
        joint ISI, or measure of independence (I believe)
    '''
    
    ## If W is just a single site, then
    try :
        N,N,K = W.shape
        
        for k in range(K) :
            W[:,:,k] = np.dot(W[:,:,k], A[:,:,k])
        
        W = np.sum(W, 2)
        
        row_sum = 0
        col_sum = 0
        
        for n in range(N) :
            row_max = np.max(W[n,:])
            col_max = np.max(W[:,n])
            
            row_sum += np.sum(W[n,:] / row_max) - 1
            col_sum += np.sum(W[:,n] / col_max) - 1
        
        tot_sum = (row_sum + col_sum) / (2 * N)
        
        return tot_sum

    ## If instead W is multiple sites, then
    except ValueError :
        N,N,K,P = W.shape
        
        for k in range(K) :
            for p in range(P) :
                W[:,:,k,p] = np.dot(W[:,:,k,p], A[:,:,k,p])
        
        W = np.sum(np.sum(W,3),2)
        
        row_sum = 0
        col_sum = 0
        
        for n in range(N) :
            row_max = np.max(W[n,:])
            col_max = np.max(W[:,n])
            
            row_sum += np.sum(W[n,:] / row_max) - 1
            col_sum += np.sum(W[:,n] / col_max) - 1
        
        ## Is it really supposed to be 2*N? I'm doubtful, think
        ## it may be something > N^2, since if after summing 
        ## all subjects get matrix of 1's, tot_sum=2N^2-2(N-1)>N^2
        tot_sum = (row_sum + col_sum) / (2 * N)
        return tot_sum