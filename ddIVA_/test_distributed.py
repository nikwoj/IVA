from local_node import local_node
from ddiva import ddiva
from master_node import master_node
from joint_isi import joint_disi, joint_isi

from numpy import zeros 
from scipy.io import savemat, loadmat
from sys import argv

def main(subj_site, num_sites) :
    master = master_node()
    
    XX = get_data_IVAG(subj_site, num_sites)
    A, wht = get_A_wht(subj_site, num_sites)
    
    P = len(XX)
    for p in range(P) :
        print "Site %d isi after IVA-G : %f" % (p, joint_isi(XX[p].W, A[p], wht[p]))
    
    avg_data = []
    for site in XX :
        avg_data.append(site.avg_data())
    
    orderings = master.order_comp(avg_data)
    for p in range(len(XX)) :
        XX[p].re_order(orderings[p])
    
    W, cost = ddiva(XX, verbose=True)
    
    #A, wht = get_A_wht(subj_site, num_sites)
    isi = joint_disi(W, A, wht)
    save_stuff_IVAG(isi, W, cost, num_sites, subj_site)
    print "(isi, num_sites, subj_per_site) : ", isi, num_sites, subj_site
    return isi

def save_stuff_IVAG(isi, W, cost, num_sites, subj_site) :
    fil = open("test_IVAG_sites" + index(num_sites) + "subj" + index(subj_site) + ".txt", "w")
    fil.write(str(isi) + ",")
    fil.write(str(num_sites) + ",")
    fil.write(str(subj_site) + "\n")
    for i in range(len(cost)) :
        fil.write(str(cost[i]) + "\n")
    
    savemat("test_IVAG_sites" + index(num_sites) + "subj" + index(subj_site) + "_W.mat", {"W":W})
    

def get_data_IVAG(subj_site, num_sites) :
    X = zeros((20, 32968, num_sites * subj_site))
    W = zeros((20, 20,    num_sites * subj_site))
    
    for k in range(num_sites) :
        W[:,:,k*subj_site : (k+1)*subj_site] = loadmat("W_IVA_G_si%d_su%d_site%d.mat" % (num_sites, subj_site, k+1))['W']
        #for kk in range(subj_site) :
        #    X[:,:,kk + subj_site*k] = loadmat("SCV_IVA_caseNik_r001_pcawhitened_subj" + index((kk+k*subj_site)+1) + ".mat")['X_white']
        X[:,:,k*subj_site : (k+1)*subj_site] = get_X_white_data(xrange(k*subj_site +1, (k+1)*subj_site +1))
    
    return [local_node(X[:,:,i*subj_site : (i+1)*subj_site], W[:,:,i*subj_site : (i+1)*subj_site]) for i in range(num_sites)]

def get_A_wht(subj_site, num_sites) :
    A   = zeros((250, 20, num_sites*subj_site))
    wht = zeros((20, 250, num_sites*subj_site))
    
    for k in range(num_sites) :
        A  [:,:,k*subj_site : (k+1)*subj_site] = get_A_data  (xrange(k*subj_site +1, (k+1)*subj_site +1))
        wht[:,:,k*subj_site : (k+1)*subj_site] = get_wht_data(xrange(k*subj_site +1, (k+1)*subj_site +1))
        #for kk in range(subj_site) :
        #    A  [:,:,kk + k*subj_site] = loadmat("A_IVA_caseNik_r001_subj" + index((kk+k*subj_site)+1) + ".mat")['A']
        #    
        #    wht[:,:,kk + k*subj_site] = loadmat("SCV_IVA_caseNik_r001_pcawhitened_subj" + index((kk+k*subj_site)+1) + ".mat")['wht']
    
    return [A[:,:,i*subj_site : (i+1)*subj_site] for i in range(num_sites)], [wht[:,:,i*subj_site : (i+1)*subj_site] for i in range(num_sites)]

def index(k) :
    if k < 10 :
        return "000%d" % k
    if k < 100 :
        return "00%d" % k
    if k < 1000 :
        return "0%d" % k
    else :
        return "%d" % k

if __name__=="__main__" :
    argv.pop(0)
    num_sites = int(argv.pop(0))
    subj_site = int(argv.pop(0))
    
    main(subj_site, num_sites)
