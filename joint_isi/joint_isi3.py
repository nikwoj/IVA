import numpy as np
from numpy import dot, abs, max, sum, zeros, identity

def joint_disi(W, Wht, A) :
    P = len(W)
    N = W[0].shape[0]
    KK = [W[p].shape[2] for p in range(P)]
    B = zeros((N,N))
    for p in range(P) :
        for k in range(KK[p]) :
            B += abs(dot(W[p][:,:,k], dot(Wht[p][:,:,k], A[p][:,:,k])))
    
    row_sum = 0
    col_sum = 0
    
    for n in range(N) :
        row_max = max(B[n,:])
        col_max = max(B[:,n])
        
        row_sum += sum(B[n,:] / row_max) - 1
        col_sum += sum(B[:,n] / col_max) - 1
    tot_sum = (row_sum + col_sum) / (2 * N * (N-1))
    return tot_sum

def joint_isi(W, A, Wht=[]) :
    N, N, K = W.shape
    if Wht == [] :
        Wht = zeros((N,N,K))
        for k in range(K) :
            Wht[:,:,k] = identity(N)
    B = zeros((N,N))
    for k in range(K) :
        B += abs(dot(W[:,:,k], dot(Wht[:,:,k], A[:,:,k])))

    row_sum = 0
    col_sum = 0

    for n in range(N) :
        row_max = max(B[n,:])
        col_max = max(B[:,n])

        row_sum += sum(B[n,:] / row_max) - 1
        col_sum += sum(B[:,n] / col_max) - 1
    tot_sum = (row_sum + col_sum) / (2 * N * (N-1))
    return tot_sum
