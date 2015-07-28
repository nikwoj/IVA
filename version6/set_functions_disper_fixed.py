import numpy as np

from numpy.linalg import pinv, det
from vec_mat import vec_to_mat, mat_to_vec


## Do we still need comp_disper? ie the parameter which tells
## us how many iterations to go before recomputing the
## dispersion matrix? The dispersion matrix is really simple 
## after all...
def set_functions(list_of_sites, verbose) :
    '''
    Initialize functions with the parameters, including the list
        of sites, the number of iterations to go without computing 
        the dispersion matrix, or whether to print output.
    '''
    
    
    
    num_subjects = 0
    initial_list_of_W = []
    K = []                          # Num subjects per site
    
    for site in list_of_sites :
        W, disper = site.initialize()
        K.append(W.shape)
        initial_list_of_W.extend( mat_to_vec(W) )
        
    P = len(K)                       # Num sites
    N, R = list_of_sites[0].shape    # Num components, samples
    KK = np.sum([a[2] for a in K])
    
    cost = 0
    
    
    def grad_and_cost ( W ) :
        if verbose :
            print "Running gradient and cost functions"
        
        # print len(W)
        # print W[0]
        W_master = []
        for p in range(P) :
            W_master.append( vec_to_mat(W, N, K[p][2]) )
        
        sqrt_YtY, list_of_disper = data_transfer_Y (W_master)
        
        cost = np.sum(sqrt_YtY)
        for p in range(P) :
            for n in range(N) :
                cost += np.log(np.abs(det(list_of_disper[p][:,:,n])))
            for k in range(K[p][2]) :
                cost = cost - np.log(np.abs(det(W_master[p][:,:,k])))
        gradient = np.array(data_transfer_grad (sqrt_YtY, W_master))
        
        return cost, gradient
    
    def data_transfer_Y ( W ) :
        ''' 
        Send W matrices, receives Y^2 values. Computes square 
            root of sum of Y^2 values, sends back to local.
            Local calculates Gradient, sends to Master.
            
            Assumes W matrices is a list of matrices.
        '''
        p = 0
        list_of_disper = []
        m_YtY = np.zeros((1,R))
        
        for site in list_of_sites :
            YtY = site.fit(W[p], dispersion)
            p += 1
            m_YtY += YtY
            list_of_disper.append(disper)
        
        return np.sqrt( m_YtY ), list_of_disper
    
    
    def data_transfer_grad ( sqrt_YtY, W ) :
        '''
        Send sqrt_YtY, receives Gradient. 
        '''
        p = 0
        gradient = []
        
        for site in list_of_sites :
            gradient.extend( mat_to_vec(site.compute_grad(sqrt_YtY, W[p]) ) )
            p += 1
        
        return gradient
    
    return N, K, P, initial_list_of_W, grad_and_cost, data_transfer_Y, data_transfer_grad
    


def set_functions2 (list_of_sites, W_init, list_of_disper=[], verbose) : 
    
    p = 0
    for site in list_of_sites :
        site._initialize2(W[p])
        p += 1
    
    def grad_and_cost ( W ) :
        if verbose :
            print "Running gradient and cost functions"
        
        # print len(W)
        # print W[0]
        W_master = []
        for p in range(P) :
            W_master.append( vec_to_mat(W, N, K[p][2]) )
        
        sqrt_YtY, list_of_disper = data_transfer_Y (W_master)
        
        cost = np.sum(sqrt_YtY)
        for p in range(P) :
            for n in range(N) :
                cost += np.log(np.abs(det(list_of_disper[p][:,:,n])))
            for k in range(K[p][2]) :
                cost = cost - np.log(np.abs(det(W_master[p][:,:,k])))
        gradient = np.array(data_transfer_grad (sqrt_YtY, W_master))
    
        return cost, gradient
    
    
    def data_transfer_grad ( sqrt_YtY, W ) :
        '''
        Send sqrt_YtY, receives Gradient. 
        '''
        p = 0
        gradient = []
        
        for site in list_of_sites :
            gradient.extend( mat_to_vec(site.compute_grad(sqrt_YtY, W[p]) ) )
            p += 1
        
        return gradient
    
    
    def data_transfer_Y ( W ) :
        ''' 
        Send W matrices, receives Y^2 values. Computes square 
            root of sum of Y^2 values, sends back to local.
            Local calculates Gradient, sends to Master.
            
            Assumes W matrices is a list of matrices.
        '''
        p = 0
        list_of_disper = []
        m_YtY = np.zeros((1,R))
        
        for site in list_of_sites :
            YtY = site.fit(W[p], dispersion)
            p += 1
            m_YtY += YtY
            list_of_disper.append(disper)
        
        return np.sqrt( m_YtY ), list_of_disper
    
    
    return grad_and_cost, data_transfer_grad, data_transfer_Y