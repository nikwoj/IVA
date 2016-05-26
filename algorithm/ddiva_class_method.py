from numpy import NaN
from local_node import local_node
from master_node import master_node


def ddiva(X, W, n_components=20, max_iter=2048, term_thresh=1e-5, verbose=False) :
    '''
    X is a list of local_node objects
        ie X = local_sites
    W is the unmixing matrix (first approx.)
    '''
    
    master = master_node()
    YtY, de_wht, Wht, KK = [], [], [], []
    cost = [NaN for it in range(max_iter)]
    al0   = 0.1
    almin = 0.1
    p = 0
    for site in X :
        stuff = site.initiate(n_components, W[p])
        YtY.append   (stuff[0])
        KK.append    (stuff[1])
        Wht.append   (stuff[2])
        de_wht.append(stuff[3])
        p += 1
    
    YtY = sum(YtY, 0)
    sqrtYtYInv = master.initiate(YtY, KK)
    backtrack = False
    
    for it in range(max_iter) :
        w_value = 0
        YtY *= 0
        for site in X :
            #local_info = site.node_step(sqrtYtYInv, al0)
            local_info = site.node_step(sqrtYtYInv, backtrack)
            YtY += local_info[0]
            w_value += local_info[1]
        
        sqrtYtYInv, cost, term = master.master_step(YtY, w_value, cost, it, verbose)

        #if it > 1 :
        #    if cost[it] > cost[it-1] : al0 = min(almin, al0*0.9)
        if it > 10 :
            if cost[it] > cost[it-1] : 
                backtrack = True
                print "backtrack"
            else : backtrack = False
        if term < term_thresh and it > 10 :
            break
    # nan = (cost[0:it] == NaN).any()
    W = []
    for site in X :
        W.append(site.finish())
    
    return W, Wht, de_wht
    # return W
    
