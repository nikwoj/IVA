from multiprocessing import Pool

from numpy import dot, zeros, identity
from numpy.random import seed, rand
from scipy.io import loadmat, savemat
from local_node import local_node
from ddiva import ddiva
from joint_isi import joint_disi

from sys import argv

# : NOTE : To use, call with 3 numbers. num_threads num_start num_stop
## num_servers : How many threads to have going at one time
## num_start : How many subjects should we start with?
## num_stop : How many subjects should we end with?
##
## If num_servers < 2 then will do a single test, and num_start will be number
## ## of subjects  to use.
## num_start should be less than num_stop. Both should be even
## All numbers should be integers
##
## Example call : python test1_d.py 20 4 180
## 
## Use 20 different threads, start with 4 subjects, end with 180 subjects
##
## Assumes data is in form 
##
##  A_IVA_caseNik_r001_subj(index).mat ['A']
##  SCV_IVA_caseNik_r001_subj(index).mat ['S']

def rem_index(k) :
    if k < 10 :
        return "000%d" % k
    if k < 100 :
        return "00%d" % k
    if k < 1000 :
        return "0%d" % k
    else : 
        return "%d" % k

def algorithm2(num_subj, Verbose=False) :
    X, A = get_data2(num_subj)
    X, A = mk_AX_data2(X, A, num_subj)
    ncomp = 20
    
    seed(0)
    W = get_W2(num_subj / 2.0, ncomp)
    W, Wht, de_wht, cost = ddiva(X, W, ncomp, verbose=Verbose)
    
    K = num_subj/2
    disi = joint_disi(W,A,Wht)
    save_stuff2(str(K), str((K,disi)), W, Wht, de_wht, cost)
    print (num_subj, disi)
    return num_subj, disi

def save_stuff2(num, write, W, Wht, de_wht, cost) :
    fil = open("test_2_ieee_" + num + ".txt"%number, "w+")
    fil.write(write + "\n")
    
    for i in range(len(cost)) :
        fil.write(str(cost[i]) + "\n")
    
    fil.close()
    savemat("test_2_ieee_" + num + "_matrices.mat", {"W":W,"Wht":Wht,"de_wht":de_wht})


def get_data2(up) :
    X = zeros((250, 32968,up))
    A = zeros((250,20,up))
    for k in range(1, up+1) :
        A[:,:,k-1] = loadmat("A_IVA_caseNik_r001_subj" + rem_index(k) + ".mat")['A']
        X[:,:,k-1] = dot(A[:,:,k-1], loadmat("SCV_IVA_caseNik_r001_subj" + rem_index(k) + ".mat")['S'])
    return X, A

def mk_AX_data2(X, A, up) :
    X_data = [local_node(X[:,:,k:k+2]) for k in range(0,up,2)]
    A_data = [A[:,:,k:k+2] for k in range(0,up,2)]
    return X_data, A_data

def get_W2(sites, ncomp) :
    sites = int(sites)
    W = [zeros((ncomp, ncomp, 2)) for k in range(sites)]
    
    for k in range(sites) :
        for kk in range(2) :
            W[k][:,:,kk] = identity(ncomp) #rand(ncomp, ncomp)
    
    return W

if __name__=="__main__" :
    if int(argv[1]) < 2 :
        subjects = int(argv[2])
        algorithm2(subjects, Verbose=True)
    else :
        PNUM = int(argv[1])

        pool = Pool(PNUM)
        
        start = int(argv[2])
        stop  = int(argv[3])
        
        B = range(start, stop, 2)
        isi = pool.map(algorithm2, B)
        fil = open("test_2_ieee_all.txt", "w+")
        for val in isi :
            fil.write(val + "\n")
        fil.close()
