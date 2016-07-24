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
    #c /= N*K
    return c

def step(it, alpha, old_norm, gW) :
    new_norm = get_norm(gW)
    if it > 0 :
        if it % 100 == 0 :
            return 1.0, new_norm
        else :
            return alpha * old_norm / new_norm, new_norm
    else :
        return 1.0, new_norm

def get_norm(gW) :
    value = np.sum(gW*gW)
    return value

def backtrack(gW, W_old, W, X, c_curr, c_prev, alpha, verbose) :
    gW_norm = get_norm(gW)
    back = 1
    while c_curr > c_prev - 1e-4 * alpha * gW_norm :
        alpha *= 0.5
        if verbose :
            print " Backtracking: %2i \t Cost: %.8f \t Alpha %.10f"%(back, alpha, c_curr)
        W = W_old + alpha * gW
        _,sqrtYtY = get_sqrtYtY(W,X)
        c_curr = cost(W,sqrtYtY)
        back += 1
    return alpha, W, c_curr, sqrtYtY


#def backtrack(gW, W, X, c_it, it, alpha, verbose) :
#    new_W = W + alpha * gW
#    _, sqrtYtY = get_sqrtYtY(new_W, X)
#    new_c = cost(new_W, sqrtYtY)
#    gW_norm = get_norm(gW)
#    back = 0
#    while new_c > c_it - 1e-4 * alpha * gW_norm :
#        alpha *= 0.5
#        new_W = W + alpha * gW
#        _, sqrtYtY = get_sqrtYtY(new_W, X)
#        new_c = cost(new_W, sqrtYtY)
#        if verbose :
#            print " Backtracking: %2i \t Cost: %.8f \t Alpha %.10f"%(back, alpha, new_c)
#        back += 1
#    dW = alpha*gW
#    return dW, alpha, new_c

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
    old_norm = 1.0
    Y, sqrtYtY = get_sqrtYtY(W,X)
    c[0] = cost(W,sqrtYtY)
    gW = grad(W,sqrtYtY,Y)
    W_old = W.copy()
    W += alpha * gW
    w_change = 0.0
    if verbose :
        print "Step: %4d  W Change: %.6f  Cost: %.6f  Alpha: %.10f"%(it,w_change,c[it],alpha)
    for it in xrange(1,max_it) :
        #dW, alpha, c[it] = backtrack(gW, W, X, c[it-1], it, alpha, verbose)
        #W_old = W.copy()
        #W += dW
        Y, sqrtYtY = get_sqrtYtY(W,X)
        c[it]      = cost(W,sqrtYtY)
        alpha, W, c[it], sqrtYtY = backtrack(gW, W_old, W, X, c[it], c[it-1], alpha, verbose)
        gW = grad(W,sqrtYtY, Y)
        alpha, old_norm = step(it, alpha, old_norm, gW)
        W_old = W.copy()
        W += alpha * gW
        #dW, alpha = backtrack(gW, W, X, c[it], it, alpha, verbose)
        w_change = term_crit(W, W_old)
        print w_change
        if w_change < term_thresh :
            break
        if verbose :
            print "Step: %4d  W Change: %.6f  Cost: %.6f  Alpha: %.10f"%(it,w_change,c[it],alpha)
    return W, c

def iva_l (X, W, verbose=True, max_it=10000, term_thresh=1e-8):
    W, cost = run_ival(X,W,verbose,term_thresh,max_it)
    return W, cost
