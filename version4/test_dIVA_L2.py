import numpy as np
import scipy.io as sio
import dIVA_L as diva
import joint_isi as isi

def main() :
    Smat = sio.loadmat("d_variables.mat")["S"]
    N,T,_ = Smat.shape
    permut = np.random.permutation([1,2,3,4,5,5])
    
    S = [[np.zeros(shape=(N,T)) for k in range(permut[p])] for p in range(len(permut))]
    
    A = [[np.random.rand(N,N)   for k in range(permut[p])] for p in range(len(permut))]
    X = [[np.zeros(shape=(N,T)) for k in range(permut[p])] for p in range(len(permut))]
    
    cc = 0
    tot =0
    while cc < 5 :
        count = 0
        
        for p in range(len(permut)) :
            for k in range(permut[p]) :
                S[p][k][:,:] = Smat[:,:,count]
                count += 1
                
                X[p][k][:,:] = np.dot(A[p][k][:,:], S[p][k][:,:])
        
        W,iteration,_ = diva.diva_l(X)
        
        a = isi.joint_ISI(W,A)
        tot += 1
        if a < .01 :
            cc += 1
        else :
            cc = cc - 2
        
        print ("Joint ISI between W and A is: %f \t Number of iterations is: %i" % (isi.joint_ISI(W,A), iteration))
        
    print tot

if __name__ == "__main__" :
    main()