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
        YtY.append(stuff[0])
        # Wht.append(stuff[1])
        KK.append (stuff[1])
        # de_wht.append(stuff[2])
        p += 1
    
    # print X[0].Y[0:3, 0:3, 0]
    
    YtY = sum(YtY, 0)
    # print "YtY"
    # print YtY
    # print X[0].W[:,:,0]
    # print X[0].X[0:3,0:3,0]
    sqrtYtYInv = master.initiate(YtY, KK)
    # print sqrtYtYInv
    for it in range(max_iter) :
        w_value = 0
        YtY *= 0
        for site in X :
            stuff = site.node_step(sqrtYtYInv, al0)
            # print stuff[0]
            YtY += stuff[0]
            w_value += stuff[1]
        
        # print "YtY"
        # print YtY
        # print "W val", w_value
        sqrtYtYInv, cost, term = master.master_step(YtY, w_value, cost, it, verbose)
        # print cost[it]
        # print sqrtYtYInv
        # break
        if it > 1 :
            if cost[it] > cost[it-1] : al0 = min(almin, al0*0.9)
        if term < term_thresh :
            break
    
    W = []
    for site in X :
        W.append(site.finish())
    
    # return W, Wht, de_wht
    return W
    