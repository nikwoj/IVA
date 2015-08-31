import numpy as np
from numpy.linalg import det

from vec_mat import vec_to_mat, mat_to_vec

def set_para ( list_of_sites, verbose ) :
    '''
    Sets the parameters to be used, as well as the functions.
        Needs to be done this way since scipy optimize works 
        best when only one parameter is to be passed to the cost
        and gradient functions. 
    '''
    
    list_of_disper = []
    list_of_YdY = []
    initial_W = []
    K = []
    
    ####
    ## Looping through sites
    ####
    for site in list_of_sites :
        initial_W.append( site.W_init )
        K.append( site.num_sub ) 
    
    ## KK is total number of subjects, N is number of components,
    ## R is number of samples, and P is number of sites
    KK = sum(K)
    N,R = site.shape
    P = len(list_of_sites)
    
    W = np.zeros((N,N,KK))
    
    count = 0
    tot   = 0
    for w in W_initial :
        W[:,:, tot:tot+K[count] ] = w
        
        count += 1
        tot   += K[count]
    
    grad = np.zeros((N,N,KK))
    
    def cost_and_grad( W_vec ) :
        '''
        Scipy optimize returns a list if passed a matrix, so we
            pass it a list and do the vectorization ourselves, 
            through the vec_mat functions
        '''
        W = vec_to_mat( W_vec, N, KK )
        
        count = 0
        tot   = 0
        ####
        ## Looping through sites
        ####
        for site in list_of_sites :
            YdY, disper = site.compute( W[:,:, tot:tot+K[count] ])
            
            list_of_YdY.append( YdY )
            list_of_disper.append( disper )
            
            count += 1
            tot   += K[count]
        
        summed_YdY = np.sqrt( sum( list_of_YdY ) )
        
        cost = np.sum( summed_YdY )
        for disper in list_of_disper :
            cost += .5 * np.log( det( disper ) )
        
        for w in W :
            cost = cost - np.log( abs( det( w ) ) )
        
        count = 0
        tot   = 0
        ####
        ## Looping through sites
        ####
        for site in list_of_sites :
            grad[:,:, tot:tot+K[count] ] = site.fit_grad( summed_YdY )
            
            count += 1
            tot   += K[count]
        
        grad_vec = mat_to_vec( grad )
        
        if verbose :
            print "Ran cost function: ", cost
        
        return cost, grad_vec
    
    return cost_and_grad, K, N, R, KK, W_initial