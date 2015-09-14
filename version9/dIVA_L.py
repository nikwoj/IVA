import numpy as np
from numpy.linalg import pinv, det
from numpy import dot

from scipy.optimize import fmin_l_bfgs_b

from vec_mat import vec_mat, mat_vec

def dIVA_L(X, W_init=[], verbose=False) :
    cost_and_grad = set_para(X, verbose)
    N,R = X[0].shape[0], X[0].shape[1]
    P   = len(K)
    K   = [x.shape[2] for x in X]
    KK  = [sum(K[0:p]) for p in range(P)]
    KK.append(0)
    
    if W_init == [] :
        W_init = [rand() for p in range(P)]
    W,d,i = fmin_l_bfgs_b
    if verbose :
        print "Optimization finished"
    W = vec_mat(W,N,KK[p-2])
    W_m = []
    for p in range(P) :
        W_m.append(W[:,:,KK[p-1]:KK[p]])
    
    return W_m, d, i

def set_para (X, verbose) :
    N,R = X[0].shape[0], X[0].shape[1]
    P   = len(X)
    K   = [x.shape[2] for x in X]
    KK  = [sum(K[0:p]) for p in range(P)]
    KK.append(0)
    W_m    = []
    disper = []
    Y = []

    def cost_and_grad(W) :
        W = vec_mat(W,N,KK[-2])
        for p in range(P) :
            W_m.append(W[:,:,KK[p-1]:KK[p])
            for k in range(K[p]) :
                Y.append(dot(W_m[p][:,:,k], X[p][:,:,k]))
        #rm(W)
        disper = [np.array([dot(Y[p][n,:,:].T, Y[p][n,:,:]) for n in range(N)]) for p in range(P)]
        A = [np.array([dot(pinv(disper[p][n,:,:]), Y[p][n,:,:]) for n in range(N)]) for p in range(P)]
        YdY = np.zeros((N,R))
        for n in range(N) :
            YdY[n,:] = sum([dot(Y[p][n,:,:].T, A[p][n,:,:]) for p in range(P)])
        YdY = np.sqrt(YdY)
        cost = np.sum(YdY) * (np.sqrt(R-1)/R)
        if verbose :
            print cost
        for p in range(P) :
            for k in range(K[p]) :
                cost = cost - log(abs(det(W_m[p][:,:,k])))
            for n in range(N) :
                cost = cost + log(det(disper[p][n,:,:])) * (1.0/2.0)
        gradient = [np.zeros((N,N,K[p])) for p in range(P)]
        for p in range(P) :
            for n in range(N) :
                B = A[p][n,:,:] * ((sqrt(R-1)/R)) / YdY )
                C = np.identity(K[p]) - dot(B, Y[p][n,:,:])
                val = B + dot(C, A[p][n,:,:])
                for k in range(K[p]) :
                    gradient[p][n,:,k] = np.dot(val[k,:], X[p][:,:,k])
            for k in range(K[p]) :
                gradient[p][:,:,k] = gradient[p][:,:,k] - pinv(W_m[p][:,:,k].T)
        gradient = mat_to_vec(gradient)
        return cost, gradient
    return cost_and_grad
