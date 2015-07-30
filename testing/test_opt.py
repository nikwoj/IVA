from optimize import iva_l
from joint_isi import joint_ISI

from scipy.io import loadmat
from autograd.numpy.random import rand
from autograd.numpy.linalg import pinv

import autograd.numpy as np

def main( scale, test_initital, optimizer ) :
    S = loadmat("d_variables.mat")['S'][:,0:200,:]
    N,R,K = S.shape
    K = 5
    X = np.zeros((N,R,K))
    A = rand(N,N,K)
    
    for k in range(K) :
        X[:,:,k] = np.dot(A[:,:,k], S[:,:,k])
    
    try :
        scale = float(scale)
        W = np.zeros((N,N,K))
        for k in range(K) :
            if test_initial :
                if k == K-1 :
                    W[:,:,k] = rand(N,N)
                else :
                    W[:,:,k] = pinv(A[:,:,k]) + scale * rand(N,N)
    except :
        W = rand(N,N,K)
    
    W, descrip = iva_l(X, W_init=W, verbose=True, optimize=optimizer)
    
    print descrip
    
    print "Joint ISI is: %f"%joint_ISI(W,A)




if __name__ == "__main__" :
    scale = raw_input("Scale factor away from inverse of A? ")
    
    try :
        scale = float(scale)
        test_initial = raw_input("Test how giving correct unmixing matrix to all but one affects output? True, False (defaults to False): ")
        try :
            test_initial = bool(test_initial)
        except :
            test_initial = False
    except :
        scale = None
        test_initial = False
    
    while True :
        optimize = raw_input('''What method do you want to optimize by? Choices are BFGS, CG, Newton-CG: ''')
        if optimize == "BFGS" :
            break
        elif optimize == "CG" :
            break
        elif optimize == "Newton-CG" :
            break
        elif optimize == "" :
            optimize = "BFGS"
            break
        else :
            print "Optimize needs to be one of the choices or empty for default (BFGS)"
    main( scale, test_initial, optimize )
