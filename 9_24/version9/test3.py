import numpy as np
import matplotlib.pyplot as plt

from dIVA_L import dIVA_L
from joint_isi import joint_ISI

def test3(X, A, verbose=False, W=[]) :
    '''
    Fix total number of subjects, increase sites and number of subjects per site
    Iterations in this test refers to number of sites
    '''
    N,R,K = X.shape
    values = []
    if W == [] :
        W = np.random.rand(N,N,K)
    KK = [[k*np.ceil(K/float(ns)) for k in range(ns+1)] for ns in range(1, np.int(np.floor(np.sqrt(K))+1))]
    print KK
    iteration = 0
    for k in KK :
        iteration += 1
        X_m = [X[:,:,k[p]:k[p+1]] for p in range(len(k)-1)]
        W_m = [W[:,:,k[p]:k[p+1]] for p in range(len(k)-1)]
        A_m = [A[:,:,k[p]:k[p+1]] for p in range(len(k)-1)]
        W_final,_,_ = dIVA_L(X_m, W_m, verbose)
        
        values.append((iteration, joint_ISI(W_m, A_m)))
    
    ISI_values = [value_pair[1] for value_pair in values]
    X_values   = [value_pair[0] for value_pair in values]
    
    plt.plot(X_values, ISI_values)
    plt.title("ISI value by number of sites with total number of subjects fixed")
    plt.xlabel("Number of sites")
    plt.ylabel("ISI value")
    plt.savefig("test3_picture")
    plt.close("all")
    
    fil = open("test3_isi_value.csv", "w")
    fil.write("ISI_value,Number_sites\n")
    for i in range(len(values)) :
        fil.write("%f,%f\n"%(ISI_values[i], X_values[i]))
    fil.close()
