from scipy.optimize import minimize
import numpy as np
import ipdb

def get_sqrtYtY(W, X):
    Y = X*0
    for k in range(X.shape[-1]):
        Y[:,:,k] = np.dot(W[:,:,k], X[:,:,k])
    sqrtYtY = np.sqrt(np.sum(Y*Y, 2))
    return Y, sqrtYtY

def grad(W, sqrtYtY, Y, R):
    gW = W*0
    N,_,K = W.shape    
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

def sole_cost(W, X):
    N,R,K = X.shape
    W = W.reshape(N,N,K)
    _, sqrtYtY = get_sqrtYtY(W, X)
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

def RMS(M,eps = 1e-3):
    return np.sqrt(M+eps)

def adadelta_optimizer(dW, decay, eps, Eg2, Ex2):
        
    Eg2 = decay * Eg2 + (1 - decay) * dW * dW
    xW = - RMS(Ex2)/RMS(Eg2) * dW
    Ex2 = decay * Ex2 + (1 - decay) * xW * xW
    
    return  xW, Eg2, Ex2


def gd_optimizer(dW, eta):
    return - eta * dW
    
def min_ival(X, W, iter = 100, decay = 0.99, eps=1e-3,
                 eta = 0.01, verbose=True):

    N,R,K = X.shape # grab parameters
    # initialize
    #np.random.seed(0)
    #W = np.random.rand(20,20,4)
    Eg2 = W * 0.0 
    Ex2 = W * 0.0
    
    for i in range(iter):
        Y, sqrtYtY = get_sqrtYtY(W, X)
        gW = grad(W, sqrtYtY, Y, R)
        c  = cost(W, sqrtYtY, R)
        
        xW, Eg2, Ex2 = adadelta_optimizer(gW, decay, eps, Eg2, Ex2)
        #xW  = gd_optimizer(gW, eta)
        
        W += xW
        if verbose:
            print '%03d cost %0.03f'%(i,c)
    return W, c
    

def iva_l (W, X, verbose=True, term_threshold=1e-8):
    results = minimize(cost_grad, W.flatten(), args=(X,), method="SLSQP",
                           jac=True, options={'disp':verbose})

    if verbose:
        if results['success'] == False :
            print "Did not converge"

    W = results['x'].reshape(W.shape)
    c = results['fun']
    return c, W
