import autograd.numpy as np
from autograd.numpy.linalg import det, pinv
from autograd.numpy.random import rand

from autograd import grad

def _get_dispersion(W,X) :
    '''
    
    '''
    N,_,K  = X.shape
    disper = np.zeros((K, K, N))
    for n in range(N) :
        value = np.dot(np.dot(W[n,:,:].T, X[n,:,:].T), np.dot(W[n,:,:], X[n,:,:])) / (N+1)
        disper[:,:,n] = value
    
    return disper



def set_para (X, W_init) :
    N,R,K = X.shape
    Y = X.copy()
    
    if W_init == [] :
        W = rand(N,N,K)
    else :
        W = W_init
    
    def compute_cost(W) :
        # for k in range(K) :
        #     Y.append(np.dot(W[:,:,k], X[:,:,k]))
        
        disper = _get_dispersion(W,X)
        cost   = 0
        # YtY    = np.zeros((N,R))
        YtY = []
        for n in range(N) :
            YtY.append(np.sum(np.dot(X[n,:,:].T, W[n,:,:].T) * np.dot(pinv(disper[:,:,n]), np.dot(X[n,:,:].T, W[n,:,:].T)), 0))
        
        for k in range(K) :
            cost += cost + np.log(np.abs(np.linalg.det(W[:,:,k])))
        
        sqrtYtY = np.sum(np.sqrt(sum(YtY)))
        cost = sqrtYtY / R - cost
        cost = cost / (N * K)

        return cost
    
    return compute_cost, W


def main() :
    X = rand(10,100,20)
    W = rand(10,10,20)
    cost, W = set_para(X, W)
    
    gradient = grad(cost)
    
    print cost(W) 
    
    print gradient(W)

if __name__ == "__main__" :
    main()