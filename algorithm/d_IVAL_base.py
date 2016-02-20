import numpy as np
from numpy import transpose, dot, zeros, NaN, isinf, sum, sqrt, log, isnan
from numpy.random import rand
from numpy.linalg import det

##############

def L_get_para(X, max_iter) :
    P = len(X)
    KK = [X[p].shape[2] for p in range(P)]
    N,T,_ = X[0].shape
    cost = [NaN for x in range(max_iter)]
    return P, KK, N, T, cost


def set_para_YtY(N, T, KK, P) :
    Y = [zeros((N,T,KK[p])) for p in range(P)]
    
    def L_YtY(X, W) : 
        YtY = zeros((N,T))
        for p in range(P) :
            for k in range(KK[p]) :
                Y[p][:,:,k] = dot(W[p][:,:,k], X[p][:,:,k])
            
            YtY += sum(Y[p]*Y[p], 2)
        return Y, YtY
    
    return L_YtY


def set_para_dW(N, T, KK, P) :
    dW = [zeros((N,N,KK[p])) for p in range(P)]
    
    def L_dW(W, Y, sqrtYtYInv) :
        for p in range(P) :
            for k in range(KK[p]) :
                phi = sqrtYtYInv * Y[p][:,:,k]
                dW[p][:,:,k] = W[p][:,:,k] - dot( dot(phi, transpose(Y[p][:,:,k])), W[p][:,:,k]) / T
        return dW
    
    return L_dW




def set_para_cost(N, T, KK, P) :
    
    def M_cost(W, sqrtYtY) :
        cost = np.sum(sqrtYtY) / T
        for p in range(P) :
            for k in range(KK[p]) :
                cost -= log(abs(det(W[p][:,:,k])))
        
        cost /= (N*sum(KK))
        return cost
    
    return M_cost



def M_terminate(cost, it, term_thresh, verbose) :
    if it < 10 :
        return False
    else :
        value = abs(cost[it-1] - cost[it]) / abs(cost[it])
        if verbose : 
            print "Iteration %d \t Cost %f \t Term Value %f" % (it, cost[it], value)
        return (value < term_thresh)

##############

def iva_l(X, alpha0=1, term_thresh=1e-6, term_crit="cost", max_iter=1024, W=[], verbose=False) :
    
    P, KK, N, T, cost = L_get_para(X, max_iter)
    L_dW   = set_para_dW(N, T, KK, P)
    L_YtY  = set_para_YtY(N, T, KK, P)
    M_cost = set_para_cost(N, T, KK, P)
    alpha_min   = 0.1
    alpha_scale = 0.9
    
    if W == [] :
        W = [rand(N,N,KK[p]) for p in range(P)]
    
    for it in range(max_iter) :
        # NOTE: Y is only used in one local function
        Y, YtY = L_YtY(X, W)
        
        sqrtYtY    = sqrt(YtY)
        sqrtYtYInv = 1/sqrtYtY
        
        dW = L_dW(W, Y, sqrtYtYInv)
        for p in range(P) :
            W[p] += alpha0 * dW[p]
        
        cost[it] = M_cost(W, sqrtYtY)
        
        term = M_terminate(cost, it, term_thresh, verbose)
        if term == True :
            break
        
        if cost[it] > cost[it-1] :
            alpha0 = min(alpha_min, alpha0*alpha_scale)
        if isinf(cost[it]) :
            W = [rand(N,N,KK[p]) for p in range(P)]
        elif isnan(cost[it]) :
            W = [rand(N,N,KK[p]) for p in range(P)]
    
    return W
    