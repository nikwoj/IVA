import numpy as np

from dIVA_L import dIVA_L

from joint_isi import joint_ISI

import matplotlib.pyplot as plt

def test1(X, A, subj_per_site, verbose=False, W=[]) :
    '''
    Fix subject per site, increase number of sites
    '''
    N,R,K = X.shape
    assert K >= subj_per_site
    X_m = []
    W_m = []
    A_m = []
    values = []
    if W == [] :
        W = np.random.rand(N,N,K)
    K = [subj_per_site * num_sites for num_sites in range(1, K / subj_per_site + 1)]
    K.append(0)
       
    for iteration in range( len(K) - 1 ) :
        X_m.append(X[:,:,K[iteration-1]:K[iteration]])
        W_m.append(W[:,:,K[iteration-1]:K[iteration]])
        A_m.append(A[:,:,K[iteration-1]:K[iteration]])
        W_final,_,_ = dIVA_L(X_m, W_m, verbose)
        values.append( (iteration+1, joint_ISI(W_final, A_m)) )
    
    ISI_values = [value_pair[1] for value_pair in values]
    X_values   = [value_pair[0] for value_pair in values]
    
    plt.plot(X_values, ISI_values)
    plt.title("ISI Value by Number of Sites")
    plt.xlabel("Number of Sites")
    plt.ylabel("ISI Value")
    plt.savefig("test1_picture")
    plt.close("all")
    
    fil = open("test1_isi_val.csv", "w")
    fil.write("ISI_value,Number_sites\n")
    for i in range(len(values)) :
        fil.write("%f,%f\n"%(values[i][1], values[i][0]))
    fil.close()

