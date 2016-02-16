import numpy as np
from numpy.linalg import pinv, det
from sys import float_info

def adjusted_IVA_L(X, KK, W_init=[], verbose=False) :
    N,R,K = X.shape
    Y = X.copy()
    
    if W_init == [] :
        W_init = np.random.rand(N,N,K)
    
    
    W_master = np.zeros((N,N,K))
    gradient = np.zeros((N,N,K))
    disper   = np.zeros((K,K,N))
    A        = np.zeros((K,R,N))
    sqrt_YtY = np.zeros((N,R))
    
    ####################################
    ####################################
    # Zeroing factor simulates distributed dispersion matrix
    #zeroing_factor = np.ones((K,K))
    #for i in range(len(KK) - 2) :
    #    zeroing_factor[KK[i-1] : KK[i],   KK[i]   : KK[i+1]] = 0
    #    zeroing_factor[KK[i]   : KK[i+1], KK[i-1] : KK[i]]   = 0
    ####################################
    ####################################
    
    
    def cost_and_grad ( W ) :
        W_master = W
        for k in range(K) :
            Y[:,:,k] = np.dot(W_master[:,:,k], X[:,:,k])
        
        for n in range(N) :
            for i in range(len(KK) -2) :
                disper[KK[i-1]:KK[i], KK[i-1]:KK[i], n] = np.dot(Y[n,:,KK[i-1]:KK[i]].T, Y[n,:,KK[i-1]:KK[i]])
            A[:,:,n]      = np.dot(pinv(disper[:,:,n]), Y[n,:,:].T)
            sqrt_YtY[n,:] = np.sqrt( np.sum( Y[n,:,:].T * A[:,:,n], 0 ) )
        
        
        ## Computing cost
        cost = np.sum( sqrt_YtY ) * (np.sqrt(R-1) / R)
        print cost
        for n in range(N) :
            cost += .5 * np.log(det(disper[:,:,n]))
            # print cost
        
        for k in range(K) :
            cost = cost - np.log(np.abs(det(W_master[:,:,k])))
            # print cost
        
        print cost, "\n\n"
        
        ## Computing Gradient
        gradient = np.zeros((N,N,K))
        for n in range(N) :
            B = A[:,:,n] * ((np.sqrt(R-1) / R) / (float_info.epsilon + sqrt_YtY[n,:]))
            C = np.identity(K) - np.dot(B, Y[n,:,:])
            value = B + np.dot(C, A[:,:,n])
            
            for k in range(K) :
                gradient[n,:,k] = np.dot(value[k,:], X[:,:,k].T)
        
        for k in range(K) :
            gradient[:,:,k] = gradient[:,:,k] - pinv(W_master[:,:,k]).T
        

        if verbose :
            print cost
            print np.linalg.norm(gradient)
        
        return cost, gradient
    
    return cost_and_grad, W_init
