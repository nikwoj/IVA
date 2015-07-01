import numpy as np

def _produce_W (W) :
    '''
    Takes in an unmixing matrix W and outputs a W such 
        that all components are grouped together.
    '''
    N,N,K = W.shape
    new_W = np.zeros(shape=(N*K, N))
    
    for n in range(N) :
        for k in range(K) :
            new_W[(n*K)+k,:] = W[:,n,k]
    
    return new_W
    
def _produce_A(A) :
    '''
    Takes in mixing matrix A and outputs an A such that
        all components are grouped together.
    '''
    N,N,K = A.shape
    new_A = np.zeros(shape=(N, N*K))
    
    for n in range(N) :
        for k in range(K) :
            new_A[:,(n*K)+k] = A[n,:,k]
    
    return new_A

def joint_ISI (W,A) :
    '''
    Takes in unmixing and mixing matrix and computes the
        joint ISI, or measure of independence (I believe)
    '''
    
    new_W = _produce_W(W)
    new_A = _produce_A(A)
    
    product = np.dot(new_W, new_A)
    
    N = product.shape[0]
    
    row_sum = 0
    col_sum = 0
    
    for n in range(N) :
        row_max = np.max(abs(product[n,:]))
        col_max = np.max(abs(product[:,n]))
        
        row_sum += np.sum(product[n,:] / row_max) - 1
        col_sum += np.sum(product[:,n] / col_max) - 1
    
    tot_sum = (row_sum + col_sum) / (2 * N)
    
    return tot_sum