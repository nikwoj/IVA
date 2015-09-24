from dIVA_L import dIVA_L

import numpy as np

from joint_isi import joint_ISI
from scipy.io import loadmat

def main() :
    S = np.zeros((16,32968,4))
    X = S.copy()
    Xm = []
    A = np.random.rand(16,16,4)
    for i in range(4) :
        S[:,:,i] = loadmat("SCV_IVA_case12_r001.mat")['Sgt'][0,i][:,:]
        X[:,:,i] = np.dot(A[:,:,i], S[:,:,i])
    Xm.append(X[:,:,0:2])
    Xm.append(X[:,:,2:4])
    joint = 1
    while joint>0.1 :
        W,_,_ = dIVA_L(Xm)
        joint = joint_ISI(W, A)
        print joint
    
main()        
