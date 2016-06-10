# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
from numpy import dot, zeros
from numpy.random import rand
from IVA_L import iva_l
# from adjusted_IVA_L import a_IVA_L
from joint_isi import joint_isi
from scipy.io import loadmat

import sys

# <codecell>

def test(fil, typ) :
    # fil is a single file
    if typ == "SCV" :
        data = loadmat(fil)['Sgt']
        X, A = make_data_SCV(data)
    elif typ == "NOR" :
        data = loadmat(fil)['S']
        X, A = make_data_NOR(data)
    
    W, _ = iva_l(X, verbose=True)
    
    print joint_ISI(W, A)

def make_data_NOR(data) :
    N,R,K = data.shape
    X = zeros((N,R,K))
    A = rand(N,N,K)
    for k in range(K) :
        X[:,:,k] = dot(A[:,:,k], data[:,:,k])
    
    return X, A

def make_data_SCV(data) :
    K = data.shape[1]
    N, R = data[0,0].shape
    
    S = zeros((N,R,K))
    X = S.copy()
    A = rand(N,N,K)
    for k in range(K) :
        S[:,:,k] = data[0,k][:,:]
        X[:,:,k] = dot(A[:,:,k], S[:,:,k])
    
    return X, A

# <codecell>

if __name__ == "__main__" :
    test(sys.argv[2], sys.argv[1]) #argv[2] is the file, argv[1] is what type it is (ie SCV?)

