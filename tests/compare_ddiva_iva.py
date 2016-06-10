import numpy as np
from numpy import dot, identity, zeros
from numpy.random import seed

from scipy.io import loadmat

#from ddiva_class_method2compare import ddiva
#from local_node import local_node
from ddiva2 import ddiva
from local_node2 import local_node

from IVA_L_5 import iva_l_4

def main() :
    NN = 250
    N  = 20
    R  = 32968
    K  = 4
    X  = zeros((NN,R,K))
    A  = zeros((NN,N,K))
    for k in range(K) :
        A[:,:,k] = loadmat("A_IVA_caseNik_r001_subj000%d.mat"%(k+1))['A']
        X[:,:,k] = dot(A[:,:,k], loadmat("SCV_IVA_caseNik_r001_subj000%d.mat"%(k+1))['S'])
    
    W = zeros((N,N,K))
    for k in range(K) :
        W[:,:,k] = identity(N)
    
    X2 = X.copy()
    X2 = [local_node(X2[:,:,0:2].copy()), local_node(X2[:,:,2:4].copy())]
    W2 = [W[:,:,0:2].copy(), W[:,:,2:4].copy()]
    
    seed(0)
    print "ddiva"
    #W2finish, _, de_wht_cost = ddiva(X2, W2, n_components=20, verbose=True)
    YtY1, cost1 = ddiva(X2,W2,n_components=20,verbose=True)
    #cost1 = de_wht_cost[1]
    
    seed(0)
    print "iva"
    #Wfinish, _, de_wht_cost = iva_l_4(X, W_init=W, n_components=20, verbose=True)
    YtY2, cost2 = iva_l_4(X,W_init=W,n_components=20,verbose=True)
    #cost2 = de_wht_cost[1]
    #Wcompare = Wfinish - W2finish
    
    #print cost1
    #print cost2
    print YtY1 - YtY2
    for i in range(2048) :
        if abs(cost1[i] - cost2[i]) > 1e-6 : print i

if __name__ == "__main__" :
    main()
