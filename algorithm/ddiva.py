from numpy import NaN, zeros
from local_node2 import local_node
from master_node2 import master_node


def ddiva(X, W, n_components=20, max_iter=2048, term_thresh=1e-6, verbose=False) :
    '''
    X is a list of local_node objects
        ie X = local_sites
    W is the unmixing matrix (first approx.)
    '''
    
    master = master_node()
    YtY, de_wht, Wht, KK = [], [], [], []
    cost = [NaN for it in range(max_iter)]
    p = 0
    for site in X :
        stuff = site.initiate(n_components, W[p])
        N = stuff[0]
        R = stuff[1]
        KK.append    (stuff[2])
        Wht.append   (stuff[3])
        de_wht.append(stuff[4])
        p += 1
    
    master.initiate(KK)
    backtrack = False
    
    for it in range(max_iter) :
        w_value = 0
        YtY = zeros((N,R))
        
        for site in X :
            local_info = site.local_step()
            YtY += local_info[0]
            w_value += local_info[1]
        
        sqrtYtYInv, backtrack, cost, term = master.master_step(YtY, w_value, cost, it, verbose)
        
        
        for site in X :
            site.local_step2(sqrtYtYInv, backtrack)
        
        #if it > 1 :
        #    if backtrack == True :
        #        if cost[it] > base :
        #            continue
        #        else :
        #            backtrack = False
        #    elif backtrack == False :
        #        if cost[it] > base :
        #            backtrack = True
        #        elif cost[it] <= base :
        #            base = cost[it]
        #else :
        #    base = cost[it]
        
        if term < term_thresh :
            break
    
    W = []
    for site in X :
        W.append(site.finish())
    
    return W, Wht, [de_wht, cost] 
