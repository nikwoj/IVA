from numpy import 

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

def step(it, alpha, old_norm, new_norm) :
    if it > 0 :
        if it % 100 == 0 :
            return 1.0
        else :
            return alpha * old_norm / new_norm
    else :
        return 1.0

def get_norm(gW) :
    K = gW.shape[2]
    value = 0.0
    for k in range(K) :
        value += np.linalg.norm(gW[:,:,k])
    return value

def backtrack(gW, W, X, c_it, alpha) :
    new_W = W + alpha * gW
    _, sqrtYtY = get_sqrtYtY(new_W, X)
    new_c = cost(new_W, sqrtYtY)
    gW_norm = get_norm(gW)
    back = 0
    while new_c > c_it - 1e-4 * alpha * gW_norm :
        alpha *= 0.5
        new_W = W + alpha * gW
        _, sqrtYtY = get_sqrtYtY(new_W, X)
        new_c = cost(new_W, sqrtYtY)
        if verbose :
            print " Backtracking: %i \t Alpha %.10f \t Cost: %f" %(back, alpha, new_c)
        back += 1
    dW = alpha*gW
    w_change = term_crit(W, W_old)
    return dW, w_change

def term_crit(W, W_old) :
    change = 0
    K = W.shape[2]
    tmp_W = W - W_old
    for k in xrange(K) :
        change = max(change, norm(tmp_W[:,:,k]))
    return change

def run_ival(X, W, verbose, term_thresh, max_it) :
    c = [np.NaN for it in xrange(max_it)]
    alpha    = 1.0
    new_norm = 1.0
    for it in xrange(max_it) :
        Y, sqrtYtY = get_sqrtYtY(W,X)
        c[it]      = cost(W, sqrtYtY, R)
        gW         = grad(W, sqrtYtY, Y)
        
        old_norm = new_norm
        new_norm = get_norm(gW)
        alpha    = step(it, alpha, old_norm, new_norm)
        dW, w_change = backtrack(gW, W, X, c, it, alpha)
        W += dW
        if w_change < term_thresh :
            break
        if verbose :
            print "Step: %d  W Change: %.6f  Cost: %.6f  Alpha: %.10f"%(it,w_change,c[it],step)
    return W, c

def iva_l (X, W, verbose=True, max_it=10000, term_thresh=1e-8):
    W, cost = run_ival(X,W,verbose,term_thresah,max_it)
    return W, cost
