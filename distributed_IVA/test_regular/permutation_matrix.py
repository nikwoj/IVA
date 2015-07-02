import numpy as np

def permutation_matrix (N) :
    '''
    Creates a random permutation matrix of shape (N,N)
    '''
    permute = np.random.permutation(N) 
    per_mat = np.identity(N)
    dummy   = np.identity(N)
    
    for n in range(N) :
        per_mat[n,:] = dummy[permute[n],:]
    
    return per_mat
    
if __name__ == "__main__" :
    
    print "Printing an example 5 dimensional permutation matrix \n", permutation_matrix(5)
    