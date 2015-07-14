import numpy as np
import scipy.io as sio

import master_node as mas
import local_node  as loc
import joint_isi   as isi


S = sio.loadmat('d_variables.mat')['S']
A = [np.random.rand(S.shape[0], S.shape[0]) for k in range(S.shape[2])]
X = []
for k in range(S.shape[2]) :
    X.append(np.dot(A[k], S[:,:,k]))

def main() :
    tot_sites = dict()
    for i in range(5) :
        tot_sites['site%s' %i] = loc.local_node([X[i]])
    
    master = mas.Master_Node(verbose=True)
    master.fit(tot_sites)
    
    for i in range(5) :
        print isi.joint_ISI(tot_sites['site%s' %i].W, A[i])

if __name__ == "__main__" :
    main()