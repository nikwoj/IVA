from cross_disper_mat import cross_disper_mat


## Local version of distribution
def _get_dispersion(X) :
    '''
    
    '''
    N,_,K  = X.shape
    disper = np.zeros((K, K, N))
    for n in range(N) :
        value = np.dot(X[n,:,:].T, X[n,:,:]) / (N+1)
        disper[:,:,n] = value
    
    return disper



## P = Number of Sites
## K = Number of subjects; dependent upon site
## N = Number of Componenets
## R = Number of samples

## Each dispersion matrix is an ndarray of shape (K,K,N), where K is that 
## sites number of subjects

## Part of master set_functions function, where N, num_disper and many other
## things are defined. 

def set_functions (list_of_sites) :
    
    
    
    num_subjects = 0
    for W_mat in Initial_list_of_W :
        num_subjects += W_mat.shape[2]
    
    N = Initial_list_of_W[0].shape[0]
    cost = 0
    m_disper = np.zeros((num_subjects, num_subjects, N))
    count = 0
    
    def compute_Master_disper (list_of_disper) :
        for comp_num in range(N) :
            count = 0
            ## What component are we doing?
            for k in range() :
                kk_count = 0
                ## What column do we want
                for kk in range(k) :
                    ## What row do we want
                    if kk == 0 :
                        num_sub = list_of_disper[k].shape[0]
                        m_disper[count:count+num_sub, 
                                 count:count+num_sub, 
                                 comp_num] = list_of_disper[k][:,:,comp_num]
                    
                    else :
                        num_subc = list_of_disper[k ].shape[1]
                        num_subr = list_of_disper[kk].shape[0]
                        value = cross_disper_mat(list_of_disper[kk][:,:,comp_num], 
                                                 list_of_disper[k] [:,:,comp_num])
                        m_disper[count-kk_count:count-kk_count+num_subr, 
                                 count:count+num_subc, 
                                 comp_num] = value
                        
                        m_disper[count:count+num_subc, 
                                 count-kk_count:count-kk_count+num_subr, 
                                 comp_num] = value.T
                    
                    kk_count += list_of_disper[kk].shape[0]
                count += list_of_disper[k].shape[0]
    
    def compute_cost_grad(W) :
        cost = np.sum(np.sqrt(sum(list_YtY)))
        for n in range(N) :
            cost += np.log(np.abs(det(master_disper[:,:,n])))
        
        for W_site in list_of_W :
            for k in range(W.shape[2]) :
                cost = cost - np.log(np.abs(det(W_site[:,:,k])))
        
        return cost




