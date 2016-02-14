# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
from numpy import dot, zeros
from numpy.random import rand
from IVA_L import iva_l
# from adjusted_IVA_L import a_IVA_L
from joint_isi import joint_ISI
from scipy.io import loadmat

import sys

# <codecell>

def test(fil) :
    # fil is a single file
    
    data = loadmat(fil)['Sgt']
    
    S, X, A = make_data(data)
    
    W, _ = iva_l(X, verbose=True)
    
    print joint_ISI(W, A)


def make_data(data) :
    K = data.shape[1]
    N, R = data[0,0].shape
    
    S = zeros((N,R,K))
    X = S.copy()
    A = rand(N,N,K)
    for k in range(K) :
        S[:,:,k] = data[0,k][:,:]
        X[:,:,k] = dot(A[:,:,k], S[:,:,k])
    
    return S, X, A

# <codecell>

if __name__ == "__main__" :
    test(sys.argv[1])

