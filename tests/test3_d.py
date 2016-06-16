from multiprocessing import Pool

from numpy import dot, zeros, abs, arange, identity
from numpy.random import rand, seed
from scipy.io import loadmat, savemat
from local_node import local_node
from ddiva import ddiva
from joint_isi import joint_disi

from sys import argv
from time import sleep

# : NOTE : To use, call with 3 numbers. num_threads num_start num_stop
## num_threads : How many threads to have going at one time
## num_start : How many sites should we start with?
## num_stop : How many sites should we end with?
##
## ## If num_threads < 2 then will do a single test, and num_start will be number
## ## of subjects to use, while num_stop will be number of sites to use
## 
## num_start should be less than num_stop. Both should be even
## All numbers should be integers
##
## Example call : python test3_d.py 20 4 180
## 
## Use 20 different threads, start with 4 sites, end with 180 sites, use all 1024
## ## 1024 subjects
##
## Assumes data is in form 
##
##  A_IVA_caseNik_r001_subj(index).mat ['A']
##  SCV_IVA_caseNik_r001_subj(index).mat ['S']

def main(argv) :
    if int(argv[1]) < 2 :
        subjects = int(argv[2])
        sites = int(argv[3])
        algorithm3(sites, num_subj=subjects, Verbose=True)
    else :
        PNUM = int(argv[1])
    
        pool = Pool(PNUM)
        
        start = int(argv[2])
        stop  = int(argv[3])
        
        B = arange(start,stop,2)
        isi = pool.map(algorithm3, B)
        fil = open("test_3_ieee_all.txt", "w+")
        fil.write(isi)
        fil.close()

def algorithm3(sites, num_subj=1024, Verbose=False) :
    ## Run a simulated test, experiment 3;
    ## Constant number of subjects increasing number of sites, decreasing
    ## ## number of subjects per site
    X, A = get_data3(num_subj)
    X, A = mk_AX_data3(X, A, num_subj, sites)
    ncomp = 20
    
    seed(0)
    W = get_W3(sites, num_subj, ncomp)
    W, Wht, de_wht, cost = ddiva(X, W, ncomp, verbose=Verbose)
    
    
    K = sites
    isi = joint_disi(W,A,Wht)
    save_stuff3(K, str((K,isi)), W, Wht, de_wht, cost)
    print (K, isi)
    return K, isi

def save_stuff3(number, write, W, Wht, de_wht, cost) :
    ## Save all relevant information from the given iteration
    fil = open("test_1_ieee_%d.txt"%number, "w")
    fil.write(write + "\n")
    
    for i in range(len(cost)) :
        fil.write(str(cost[i]) + "\n")
    fil.close()
    
    savemat("test_1_ieee_%d_matrices.mat"%number, {"W":W,"Wht":Wht,"de_wht":de_wht})

def get_data3(up=1024) :
    ## Get subjects 1 through up+1, for a total of up subjects
    ## Defaults to up=1024, in other words, the full dataset
    X = zeros((250, 32968, up))
    A = zeros((250, 20, up))
    for k in range(1, up+1) :
        A[:,:,k-1] = loadmat("A_IVA_caseNik_r001_subj" + rem_index(k) + ".mat")['A']
        X[:,:,k-1] = dot(A[:,:,k-1], loadmat("SCV_IVA_caseNik_r001_subj" + rem_index(k) + ".mat")['S'])
    return X, A

def mk_AX_data3(X, A, subjects, sites) :
    ## NOTE 
    ## Due to how ddiva is set up, no site can have only one subject.
    ## Thus, need to be careful of how A, X, and W get distributed
    subj = subjects / sites
    remainder = subjects % sites
    
    if remainder == 1 :
        A_data = [A[:,:, k*subj : (k+1)*subj] for k in range(sites-2)]
        X_data = [local_node(X[:,:, k*subj : (k+1)*subj]) for k in range(sites-2)]
        
        A_data.append(A[:,:, k*(subj-1) : (k+1)*(subj-1)])
        X_data.append(local_node(X[:,:, k*(subj-1) : (k+1)*(subj-1)]))
        
        A_data.append(A[:,:, -2 : ])
        X_data.append(local_node(X[:,:, -2 : ]))
        
    else :
        A_data = [A[:,:, k*subj : (k+1)*subj] for k in range(sites-1)]
        X_data = [local_node(X[:,:, k*subj : (k+1)*subj]) for k in range(sites-1)]
        
        A_data.append(A[:,:, (sites-1)*subj : ])
        X_data.append(local_node(X[:,:, (sites-1)*subj : ]))

    return X_data, A_data

def get_W3(sites, subjects, ncomp) :
    ## NOTE 
    ## Due to how ddiva is set up, no site can have only one subject.
    ## Thus, need to be careful of how A, X, and W get distributed
    subj = subjects / sites
    remainder = subjects % sites
    if remainder == 1 :
        W = [zeros((ncomp, ncomp, subj)) for x in range(sites-2)]
        W.append(zeros((ncomp, ncomp, subj-1)))
        W.append(zeros((ncomp, ncomp, 2)))
        
        for k in range(sites - 2) :
            for kk in range(subj) :
                W[k][:,:,kk] = identity(ncomp) #rand(ncomp, ncomp)
        for kk in range(subj-1) :
            W[-2][:,:,kk] = identity(ncomp)#rand(ncomp, ncomp)
        
        W[-1][:,:,0] = identity(ncomp) #rand(ncomp, ncomp)
        W[-1][:,:,1] = identity(ncomp) #rand(ncomp, ncomp)    
    
    elif remainder == 0 :
        W = [zeros((ncomp, ncomp, subj)) for k in range(sites)]
        W.append(zeros((ncomp, ncomp, remainder)))
        
        for k in range(sites) : 
            for kk in range(subj) :
                W[k][:,:,kk] = identity(ncomp) #rand(ncomp, ncomp)
    else :
        W = [zeros((ncomp, ncomp, subj)) for k in range(sites-1)]
        W.append(zeros((ncomp, ncomp, remainder)))
        
        for k in range(sites-1) : 
            for kk in range(subj) :
                W[k][:,:,kk] = identity(ncomp) #rand(ncomp, ncomp)
        for kk in range(remainder) :
            W[-1][:,:,kk] = identity(ncomp) #rand(ncomp, ncomp)
    return W

def rem_index(k) :
    if k < 10 :
        return "000%d" % k
    if k < 100 :
        return "00%d" % k
    if k < 1000 :
        return "0%d" % k
    else : 
        return "%d" % k

if __name__=="__main__" :
    main(argv)



