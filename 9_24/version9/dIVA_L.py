import numpy as np
from numpy.linalg import pinv, det
from numpy import dot, log, sqrt

from scipy.optimize import fmin_l_bfgs_b

from vec_mat import vec_to_mat, mat_to_vec

def dIVA_L(X, W_init=[], verbose=False) :
    cost_and_grad = set_para(X, verbose)
    N,R = X[0].shape[0], X[0].shape[1]
    K   = [x.shape[2] for x in X]
    P   = len(K)
    KK  = [sum(K[0:p+1]) for p in range(P)]
    KK.append(0)
    
    if W_init == [] :
        W_init = [np.random.rand(N,N,K[p]) for p in range(P)]
    W_init = mat_to_vec(W_init)
    W,d,i = fmin_l_bfgs_b(cost_and_grad, x0=W_init)
    if verbose :
        print "Optimization Finished"
    W = vec_to_mat(W,N,KK[-2])
    W_m = []
    for p in range(P) :
        W_m.append(W[:,:,KK[p-1]:KK[p]])
    
    return W_m, d, i

def set_para (X, verbose) :
    N,R = X[0].shape[0], X[0].shape[1]
    P   = len(X)
    K   = [x.shape[2] for x in X]
    KK  = [sum(K[0:p+1]) for p in range(P)]
    KK.append(0)
    
    def cost_and_grad(W) :
        print "Running"
        W = vec_to_mat(W,N,KK[-2])
        W_m = []
        disper = []
        Y = [np.zeros((N,R,k)) for k in K]
        for p in range(P) :
            W_m.append(W[:,:,KK[p-1]:KK[p]])
            for k in range(K[p]) :
                Y[p][:,:,k] = dot(W_m[p][:,:,k], X[p][:,:,k])
        disper = [np.array([dot(Y[p][n,:,:].T, Y[p][n,:,:]) for n in range(N)]) for p in range(P)]
        A = [np.array([dot(pinv(disper[p][n,:,:]), Y[p][n,:,:].T) for n in range(N)]) for p in range(P)]
        YdY = np.zeros((N,R))
        for n in range(N) :
            YdY[n,:] = np.sum(sum([Y[p][n,:,:].T * A[p][n,:,:] for p in range(P)]), 0)
            #for p in range(P) :
                #YdY[n,:] = YdY[n,:] + np.sum(Y[p][n,:,:].T * A[p][n,:,:], 0)
        YdY = np.sqrt(YdY)
        cost = np.sum(YdY) * (np.sqrt(R-1)/R)
        for p in range(P) :
            for k in range(K[p]) :
                cost = cost - log(abs(det(W_m[p][:,:,k])))
            for n in range(N) :
                cost = cost + log(det(disper[p][n,:,:])) * (1.0/2.0)
        if verbose :
            print cost
        gradient = [np.zeros((N,N,K[p])) for p in range(P)]
        for p in range(P) :
            for n in range(N) :
                B = A[p][n,:,:] * (((sqrt(R-1)/R)) / YdY[n,:] )
                C = np.identity(K[p]) - dot(B, Y[p][n,:,:])
                val = B + dot(C, A[p][n,:,:])
                for k in range(K[p]) :
                    gradient[p][n,:,k] = np.dot(val[k,:], X[p][:,:,k].T)
            for k in range(K[p]) :
                gradient[p][:,:,k] = (gradient[p][:,:,k] - pinv(W_m[p][:,:,k]).T)
        gradient = np.array(mat_to_vec( gradient ))
        return cost, gradient
    return cost_and_grad
