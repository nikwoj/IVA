from numpy import NaN, zeros
from local_node import local_node
from master_node import master_node


def ddiva(X, ncomp=0, max_iter=2048, term_thresh=1e-6, verbose=False) :
    '''
    X is a list of local_node objects
        ie X = local_sites
        Each local_node object has two instances: 
            X, the site's data
            W, the site's initial unmixing matrix
    '''
    
    master = master_node()
    Wht, de_wht, KK = [], [], []
    cost = [NaN for it in range(max_iter)]
    
    for site in X :
        stuff = site.initiate(ncomp)
        N =       stuff[0]
        R =       stuff[1]
        KK.append(stuff[2])
        if ncomp > 0 : 
            Wht.append   (stuff[3])
            de_wht.append(stuff[4])
    
    master.initiate(KK)
    backtrack = False
    
    for it in range(max_iter) :
        w_value = 0
        YtY = zeros((N,R))
        
        for site in X :
            local_info = site.local_step()
            YtY     += local_info[0]
            w_value += local_info[1]
        
        sqrtYtYInv, backtrack, cost, term = master.master_step(YtY, w_value, cost, it, verbose)
        
        for site in X :
            site.local_step2(sqrtYtYInv, backtrack)
        
        if term < term_thresh :
            break
    
    W = []
    for site in X :
        W.append(site.finish())
    
    if ncomp == 0 :
        return W, cost[0:it]
    
    if ncomp > 0 :
        return W, Wht, de_wht, cost[0:it]
