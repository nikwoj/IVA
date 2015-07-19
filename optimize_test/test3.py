import autograd.numpy as np
from autograd import grad

def h(X) :
    N,N,K = X.shape
    a = 0
    
    for k in range(K) :
        a += np.linalg.det(X[:,:,k])
    
    return a

if __name__ == "__main__" :
    cost = grad(h)
    
    print cost(np.random.rand(3,3,4))