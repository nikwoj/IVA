from scipy.optimize import minimize
import numpy as np

def iva_l (X, W, verbose) :
    N,N,K = W.shape
    W = W.reshape(N*N*K)
    cost_grad = cost_grad_wrapper(X)
    
    results = minimize(cost_grad, W, method="CG", jac=True,
                       options={'disp':verbose})
    
    if results['success'] == False : 
        if verbose : print "Did not converge"
    
    W = results['x'].reshape(N,N,K)
    return W

def cost_grad_wrapper (X) :
    N,R,K = X.shape
    Y=X*0
    dW=np.zeros((N,N,K))
    
    def cost_grad(W) :
        W = W.reshape(N,N,K)
        Y, sqrtYtY = get_sqrtYtY(X, W)
        grad = get_grad(Y, sqrtYtY, W)
        grad = grad.reshape(N*N*K)
        cost = get_cost(sqrtYtY, W)
        return cost, grad

    def get_sqrtYtY(X, W) :
        _,_,K = X.shape
        for k in range(K) :
            Y[:,:,k] = np.dot(W[:,:,k], X[:,:,k])
        sqrtYtY = np.sqrt(np.sum(Y*Y, 2))
        return Y, sqrtYtY
    
    def get_grad(Y, sqrtYtY, W) :
        sqrtYtY = 1 / sqrtYtY
        for k in range(K) :
            phi = sqrtYtY * Y[:,:,k]
            dW[:,:,k] = W[:,:,k] - np.dot(np.dot(phi, Y[:,:,k].T / R), W[:,:,k])
        print "Gradient norm : ", np.max([np.linalg.norm(dW[:,:,k], 2) for k in range(K)])
        return -1 * dW
    
    def get_cost(sqrtYtY, W) :
        cost = np.sum(sqrtYtY)/R
        for k in range(K) :
            Q,L = np.linalg.qr(W[:,:,k])
            L = np.diag(L)
            cost -= np.sum(np.log(np.abs(L)))
        
        cost /= (N*K)
        print "Cost function : ", cost
        return cost
        
    return cost_grad
