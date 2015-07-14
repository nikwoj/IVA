import numpy     as np
import scipy.io  as sio
import dIVA_L    as diva
import IVA_L     as iva
import timeit    as ti
import joint_isi as isi


if __name__ == "__main__" :
    variables = sio.loadmat('d_variables.mat')
    
    Sm = variables['S']
    N,T,K = Sm.shape
    
    print "Values printed in \n(regular, distributed) Distributed uses total number of subjects /2 number of subjets per site"
    ## We will now test the difference between regular 
    ## and distributed IVA, by comparing X subjects in
    ## IVA with X/2 sites, each with 2 subjects, in 
    ## dIVA
    
    ## Note that in this case, X=20, so want to hit 
    ## values 4, 6, ..., 18, 20 
    ## Start at 4, go up to 22 (which is not included),
    ## and proceed by 2's. Don't start at 2 since its 
    ## trivial (one site two subjects is normal IVA)
    for j in range(4, 22, 2) :
        X = np.zeros(shape=(N,T,j))
        A = np.random.rand(N,N,j)
        Z = np.random.rand(N,N,j)
        for i in range(j) :
            X[:,:,i] = np.dot(A[:,:,i], Sm[:,:,i])
        
        W, _ ,_ = iva.iva_l(X)
        # W,_,_ = iva.iva_l(X, W_init=Z)
        
        a = isi.joint_ISI(W,A)
        # print ("Joint ISI for normal IVA with %i subjects is: \n\t\t\t\t\t\t %f" 
                # % (j, isi.joint_ISI(W,A)))
        
        ## Want each site to have two subjects, like cutting 
        ## number of subjects in half.
        X = np.zeros(shape=(N, T, 2, j/2.0))
        B = np.zeros(shape=(N, N, 2, j/2.0))
        C = np.zeros(shape=(N, N, 2, j/2.0))
        for i in range(j/2) :
            
            ## 2 subjects per site
            for k in range(2) :
                C[:,:,k,i] = Z[:,:,k+2*i]
                B[:,:,k,i] = A[:,:,k+2*i]
                X[:,:,k,i] = np.dot(B[:,:,k,i], Sm[:,:,k+2*i])
        
        # W, _, _ = diva.diva_l(X, W_init=C)
        W, _, _ = diva.diva_l(X)
        
        b = isi.joint_ISI(W,B)
        
        print "(%f, \t %f), %i total number of subjects" % (a, b, j)
        # print ("Joint ISI for distributed IVA with 2 subjects per %i sites is: \n\t\t\t\t\t\t %f" 
                # %  (i+1, isi.joint_ISI(W,B)))
