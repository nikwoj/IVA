import numpy as np
import scipy.io as sio
import dIVA_L as diva
import joint_isi as isi

def main() :
    variables = sio.loadmat("d_variables.mat")
    
    Smat = variables['S']
    N,T,K = Sm.shape
    Svar = [np.zeros(shape=(N,T)) for k in range(K)]
    
    for k in range(K) :
        Svar[k][:,:] = Smat[:,:,k]
    
    ## Have 5 sites, increase number of subjects per site
    ## until have twenty total subjects
    for p in range(1,5) :
        S = [[np.zeros(shape=(N,T)) for k in range(p)] for l in range(5)]
        
        
        
        for l in range(5) :
            for k in range(p) :
                S[l][k][:,:] = Svar[l*p+k][:,:]
            
            A = [[np.random.rand(N,N)