import numpy as np
import matplotlib.pyplot as plt

from dIVA_L import dIVA_L
from joint_isi import joint_ISI

def test2(X, A, num_sites, verbose=False, W=[]) :
    '''
    Fix number of sites, increase number of subjects per site
    '''
    N,R,K = X.shape
    values = []
    if W == [] :
        W = np.random.rand(N,N,K)
    KK = [[a*p for a in range(num_sites + 1)] for p in range(1, K / num_sites + 1)]
    for iteration in range(K / num_sites) :
        X_m = []
        W_m = []
        A_m = []
        for site in range(num_sites) : 
            X_m.append(X[:,:,KK[iteration][site]:KK[iteration][site+1]])
            W_m.append(W[:,:,KK[iteration][site]:KK[iteration][site+1]])
            A_m.append(A[:,:,KK[iteration][site]:KK[iteration][site+1]])
        W_final, _, _ = dIVA_L(X_m, W_m, verbose=True)
        values.append((iteration+1, joint_ISI(W_final, A_m)))
    ISI_values = [value_pair[1] for value_pair in values]
    X_values   = [value_pair[0] for value_pair in values]
    
    plt.plot(X_values, ISI_values)
    plt.title("ISI Value by Number of Subjects per Site")
    plt.xlabel("Number of Subjects per Site")
    plt.ylabel("ISI Value")
    plt.savefig("test2_picture")
    plt.close("all")
    
    fil = open("test2_isi_val.csv", "w")
    fil.write("ISI_value,Subjects_per_site\n")
    for i in range(len(values)) :
        fil.write("%f,%f\n"%(ISI_values[i], X_values[i]))
    fil.close()
