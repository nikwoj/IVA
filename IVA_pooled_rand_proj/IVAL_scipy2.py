from scipy.optimize import minimize
import numpy as np

def get_sqrtYtY(W, X):
    Y = X*0
    for k in range(X.shape[-1]):
        Y[:,:,k] = np.dot(W[:,:,k], X[:,:,k])
    sqrtYtY = np.sqrt(np.sum(Y*Y, 2))
    return Y, sqrtYtY

def grad(W, sqrtYtY, Y, R):
    N,_,K = W.shape
    gW = W*0
    sqrtYtY = 1 / sqrtYtY
    for k in range(K):
        phi = sqrtYtY * Y[:,:,k]
        gW[:,:,k] = W[:,:,k] - np.dot(np.dot(phi, Y[:,:,k].T / R), W[:,:,k])
    return -1 * gW

def cost(W, sqrtYtY, R):
    N,_,K = W.shape
    c = np.sum(sqrtYtY)/R
    for k in range(K):
        c -= np.linalg.slogdet(W[:,:,k])[1]
    c /= N*K
    return c

def cost_grad(W, X):
    N,R,K = X.shape
    W = W.reshape(N,N,K)
    Y, sqrtYtY = get_sqrtYtY(W, X)
    gW = grad(W, sqrtYtY, Y, R).flatten()
    c = cost(W, sqrtYtY, R)
    return c, gW

def iva_l (X, W, verbose=True, term_threshold=1e-8):
    Wshape = W.shape
    results = minimize(cost_grad, W.flatten(), args=(X,), method="CG",
                           jac=True, options={'disp':verbose})

    if verbose:
        if results['success'] == False :
            print "Did not converge"

    W = results['x'].reshape(Wshape)
    print W.shape
    c = results['fun']
    return W
