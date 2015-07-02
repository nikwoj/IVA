import numpy as np

def _produce_W (W) :
    '''
    Takes in an unmixing matrix W and outputs a W such 
        that all components are grouped together.
    '''
    N,N,K,P = W.shape
    new_W   = np.zeros(shape=(N*(K+P), N))
    
    
    for n in range(N) :
        for p in range(P) :
            for k in range(K) :
                new_W[(n*(K+P))+k+p,:] = W[:,n,k,p]
    
    return new_W
    
def _produce_A(A) :
    '''
    Takes in mixing matrix A and outputs an A such that
        all components are grouped together.
    '''
    N,N,K,P = A.shape
    new_A   = np.zeros(shape=(N, N*(K+P)))
    
    for n in range(N) :
        for p in range(P) :
            for k in range(K) :
                new_A[:,(n*(K+P))+k+p] = A[n,:,k,p]
    
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
