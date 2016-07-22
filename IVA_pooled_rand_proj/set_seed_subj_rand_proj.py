from numpy import dot, zeros, sqrt, identity
from numpy.random import permutation, seed, rand, normal, randint
from numpy.linalg import qr, svd
from sys import argv
from scipy.io import savemat

from import_SCV_data import get_X_data
## Writes a file with the names of the subjects to use. Matlab will
## then read this file and thus, Matlab and python are using same data

def get_rand_proj(method) :
    if method == "normal" :
        proj = rand(20,250)
        return proj
    if method == "subspace" :
        proj = zeros((20,250))
        permute = permutation(250)[:20]
        for n in range(20) :
            proj[n,permute[n]] = 1
        return proj
    if method == "qr" :
        proj = normal(size=(250,20))
        Q,R = qr(proj, mode='raw')
        return Q
    if method == "singular_value" :
        proj = rand(20,250)
        U,S,V = svd(proj, compute_uv=True)
        S = zeros((20,250))
        S[:20,:20] = identity(20)
        return dot(U,dot(S,V))

def main (subjs, set_seed, method=False) :
    seed(0)
    #if method==False :
    #    proj = get_rand_proj("normal")
    #else :
    #    proj = get_rand_proj(method)
    proj = get_rand_proj(method)
    
    list_subjs = range(1, subjs+1)
    X = get_X_data(list_subjs)
    Y = zeros((20,32968,subjs))
    for k in range(subjs) :
        Y[:,:,k] = dot(proj, X[:,:,k])
    
    seed(set_seed)
    W = rand(20,20,subjs)
    
    #if method==False :
    #    savemat("SCV_IVA_rand_proj_W_subj%d_start%d.mat"%(subjs, set_seed), {'X':Y, 'W':W, 'proj':proj})
    #else :
    #    savemat("SCV_IVA_rand_proj_W_subj%d_start%d_method%s.mat"%(subjs, set_seed, method), {'X':Y, 'W':W, 'proj':proj})
    savemat("SCV_IVA_rand_proj_W_subj%d_start%d_method_%s.mat"%(subjs, set_seed, method), {'X':Y, 'W':W, 'proj':proj})

def index4 (k) :
    if k < 10 : return "000" + str(k)
    if k < 100 : return "00" + str(k)
    if k < 1000 : return "0" + str(k)
    else : return str(k)

if __name__ == "__main__" :
    subjs = int(argv.pop(1))
    set_seed = int(argv.pop(1))
    #try :
    #    method = argv.pop(1)
    #    main(subjs, set_seed, method)
    #except :
    #    main(subjs, set_seed)
    
    method = argv.pop(1)
    main(subjs, set_seed, method)
