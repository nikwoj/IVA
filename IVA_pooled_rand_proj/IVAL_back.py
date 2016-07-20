import numpy as np
from numpy import dot, zeros
from numpy.linalg import norm
from ica import pca_whiten

def iva_l (X, W, term_threshold=1e-6, term_crit='ChangeInW',
           max_iter=2048, A = [], verbose=False, n_components=0) :
    
    cost = [np.NaN for x in range(max_iter)]
    if n_components > 0 :
        wht = zeros((n_components, N, K))
        de_wht = zeros((N, n_components, K))
        X_white = zeros((n_components, T, K))
        for k in range(K) :
            X_white[:,:,k], wht[:,:,k], de_wht[:,:,k] = pca_whiten(X[:,:,k], n_components, verbose=verbose)
        X = X_white
    Y = X.copy()
    N,T,K = X.shape
    
    s_cons = 1e-4
    alpha = 0.1
    dW_norm = 1.0
    for it in range(max_iter) :
        
        ## Initial approximation to true source vectors
        for k in range(K) :
            Y[:,:,k] = np.dot(W[:,:,k], X[:,:,k])
        
        sqrtYtY    = np.sqrt(np.sum(Y*Y,2))
        cost[it] = compute_cost(W, sqrtYtY)
        
        if it > 0 :
            back_num = 0
            while cost[it] > cost[it-1] - s_cons * alpha * old_norm :
                alpha *= 0.5
                W = W_old.copy() + alpha * dW
                for k in range(K) :
                    Y[:,:,k] = np.dot(W[:,:,k], X[:,:,k])
                sqrtYtY = np.sqrt(np.sum(Y*Y,2))
                cost[it] = compute_cost(W, sqrtYtY)
                
                if verbose :
                    print " Backtracking: %i \t Alpha : %.10f \t  Cost: %f" % (back_num, alpha, cost[it])
                back_num += 1
                
        sqrtYtYInv = 1 / sqrtYtY
        old_norm = dW_norm
        
        dW = get_dW(W, Y, sqrtYtYInv)
        dW_norm = grad_norm(dW)
        W_old = W.copy()
        
        alpha = get_alpha(it, alpha, old_norm, dW_norm)
        W += alpha * dW
        
        ## Check termination Criterion
        if term_crit == 'ChangeInW' :
            term_criterion = 0
            for k in range(K) :
                tmp_W = W_old[:,:,k] - W[:,:,k]
                term_criterion = max(term_criterion, np.linalg.norm(tmp_W[:,:], ord=2))
        
        elif term_crit == 'ChangeInCost' :
            if it == 0 :
                term_criterion = 1.0
            else :
                term_criterion = (abs(cost[it-1]-cost[it])
                                 / abs(cost[it]))
        
        ## Check termination condition
        if term_criterion < term_threshold or it == max_iter :
            break
        ## Display iteration information
        if verbose :
            print "Step: %i \t W change: %f \t Cost %f \t Alpha%f" % (it, term_criterion, cost[it], alpha)
            ## End iteration
    
    ## Finish display
    if verbose :
        print "Algorithim converged, end results are: "
        print " Step: %i \n W change: %f \n Cost %f \n\n" % (it, term_criterion, cost[it])
    if n_components > 0 :
        return W, wht, de_wht, cost
    else :
        return W, cost[:it]

def get_alpha(it, alpha, old_norm, dW_norm) : 
    if it > 0 : return alpha * old_norm / dW_norm
    else : return 1.0

def grad_norm(dW) : 
    dW_norm = 0.0
    for k in range(dW.shape[2]) :
        dW_norm += norm(dW[:,:,k],2)
    return dW_norm

def get_dW (W, Y, sqrtYtYInv) :
    _, R, K = Y.shape
    dW = W.copy()
    
    for k in range(K) :
        phi = sqrtYtYInv * Y[:,:,k]
        dW[:,:,k] = W[:,:,k] - np.dot( np.dot(phi, np.transpose(Y[:,:,k]) / R), W[:,:,k])
    
    return dW

def compute_cost (W, sqrtYtY) :
    N, N, K = W.shape
    R = sqrtYtY.shape[1]
    current_cost = 0
    test = np.sum(sqrtYtY)
    for k in range(K) :
        Q, L = np.linalg.qr(W[:,:,k])
        L = np.diag(L)
        current_cost += np.sum(np.log(np.abs(L)))
    
    current_cost = (-1)*current_cost + np.sum(sqrtYtY) / R
    #current_cost = current_cost / (N*K)
    
    return current_cost

