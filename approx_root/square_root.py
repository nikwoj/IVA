import numpy as np

def square_root ( A, error ) :
    '''
    Takes in ndarray and outputs approximation to square root for 
        every element in A
    
    Inputs:
    -------
    A : ndarray
        ndarray that the square root element-wise is wanted
    
    Outputs:
    --------
    B : ndarray, shape = A.shape
        A square rooted elementwise
    '''
    shape = A.shape
    eps   = 1
    x     = [A / 2.0]
    
    while eps > error :
        x.append( (x[-1] + A / x[-1]) / 2.0 )
        eps = np.max(x[-1] - x[-2])
    
    return x[-1]