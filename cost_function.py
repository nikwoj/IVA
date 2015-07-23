def comp_Y  (W) : return np.array([np.dot(W[:,:,k], X[:,:,k]) for k in range(K)])
    
def comp_dis(Y) : return np.array([np.dot(Y[:,n,:], Y[:,n,:].T) for n in range(N)])
        
def sqrt(Y,dispersion) :
    return np.sum(np.sqrt(np.array([np.sum(Y[:,n,:] * np.dot(inv(dispersion [n,:,:]), Y[:,n,:]), 0)
                                        for n in range(N)])))
    
def compute_cost(W) :
    cost = 0
    Y = comp_Y(W)
    disper = comp_dis(Y)
    sqrtYtY = sqrt(Y, disper)
    for k in range(K) :
        cost = cost - np.log(np.abs(det(W[:,:,k])))
        
    cost = cost + np.sqrt(R-1) * sqrtYtY / R
    for n in range(N) :
        cost = cost + 0.5 * np.log(det(disper[n,:,:])))
    
    return cost

N,R,K = X.shape