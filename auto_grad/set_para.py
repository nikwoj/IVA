import autograd.numpy as np
from autograd.numpy.linalg import inv, det

def set_para (X, W_init) :
    N,R,K = X.shape
    Y = X.copy()
    
    if W_init == [] :
        W = rand(N,N,K)
    else :
        W = W_init
    
    def comp_Y  (W) : return np.array([np.dot(W[:,:,k], X[:,:,k]) for k in range(K)])
    
    def comp_dis(Y) : return np.array([np.dot(Y[:,n,:], Y[:,n,:].T) for n in range(N)])
        
    def sqrt(Y,dispersion) :
        return np.sum(np.sqrt(np.array([np.sum(Y[:,n,:] * np.dot(inv(dispersion[n,:,:]), Y[:,n,:]), 0)
                                        for n in range(N)])))
    
    def compute_cost(W) :
        cost = 0
        Y = comp_Y(W)
        disper = comp_dis(Y)
        sqrtYtY = sqrt(Y, disper)
        for k in range(K) :
            cost = cost - np.log(np.abs(det(W[:,:,k])))
        
        cost = cost + np.sqrt(R-1) / R * sqrtYtY
        for n in range(N) :
            cost = cost + 0.5 * np.log(np.abs(det(disper[n,:,:])))
        return cost
    
    return compute_cost, W
    
