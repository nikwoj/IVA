import numpy as np

def get_sqrtYtY(W, X):
    Y = X*0
    for k in range(X.shape[-1]):
        Y[:,:,k] = np.dot(W[:,:,k], X[:,:,k])
    sqrtYtY = np.sqrt(np.sum(Y*Y, 2))
    return Y, sqrtYtY

def grad(W, sqrtYtY, Y):
    N,_,K = W.shape
    R = sqrtYtY.shape[1]
    gW = W*0
    sqrtYtY = 1 / sqrtYtY
    for k in range(K):
        phi = sqrtYtY * Y[:,:,k]
        gW[:,:,k] = W[:,:,k] - np.dot(np.dot(phi, Y[:,:,k].T / R), W[:,:,k])
    return gW

def cost(W, sqrtYtY):
    K = W.shape[2]
    R = sqrtYtY.shape[1]
    c = np.sum(sqrtYtY)/R
    for k in range(K):
        c -= np.linalg.slogdet(W[:,:,k])[1]
    return c

def step(it, alpha, old_norm, gW) :
    new_norm = get_norm(gW)
    if it % 100 == 99 :
        return 1.0, new_norm
    else :
        return alpha * old_norm / new_norm, new_norm

def get_norm(gW) :
    value = np.sum(gW*gW)
    return value

def backtrack(gW, W_old, W, X, c_curr, c_prev, alpha, Y, sqrtYtY, verbose) :
    gW_norm = get_norm(gW)
    back = 1
    while c_curr > c_prev - 1e-16 * alpha * gW_norm :
        alpha *= 0.5
        if verbose :
            print " Backtracking: %2i \t Cost: %.8f \t Alpha %.30f"%(back, c_curr, alpha)
        W = W_old + alpha * gW
        Y, sqrtYtY = get_sqrtYtY(W,X)
        c_curr = cost(W, sqrtYtY)
        back += 1
    return alpha, W, c_curr, sqrtYtY, Y

def term_crit(W, W_old) :
    change = 0
    K = W.shape[2]
    tmp_W = W - W_old
    for k in xrange(K) :
        change = max(change, np.linalg.norm(tmp_W[:,:,k]))
    return change

def run_ival(X, W, verbose, term_thresh, max_it) :
    c = [np.NaN for it in xrange(max_it)]
    alpha    = 1.0
    Y, sqrtYtY = get_sqrtYtY(W,X)
    c[0] = cost(W,sqrtYtY)
    gW = grad(W,sqrtYtY,Y)
    old_norm = get_norm(gW)
    W_old = W.copy()
    W += alpha * gW
    w_change = 0.0
    if verbose :
        print "Step: %4d  W Change: %.6f  Cost: %.6f  Alpha: %.30f"%(it,w_change,c[it],alpha)
    for it in xrange(1,max_it) :
        Y, sqrtYtY = get_sqrtYtY(W,X)
        c[it]      = cost(W,sqrtYtY)
        alpha, W, c[it], sqrtYtY, Y = backtrack(gW, W_old, W, X, c[it], c[it-1], 
                                                alpha, Y, sqrtYtY, verbose)
        gW = grad(W, sqrtYtY, Y)
        alpha, old_norm = step(it, alpha, old_norm, gW)
        W_old = W.copy()
        W += alpha * gW
        w_change = term_crit(W, W_old)
        if w_change < term_thresh :
            break
        if verbose :
            print "Step: %4d  W Change: %.6f  Cost: %.6f  Alpha: %.30f"%(it,w_change,c[it],alpha)
    return W, c

def iva_l (X, W, verbose=True, max_it=10000, term_thresh=1e-8):
    W, cost = run_ival(X,W,verbose,term_thresh,max_it)
    return W, cost
