import numpy as np
import matplotlib.pyplot as plt

from joint_isi import joint_ISI
from local_node import local

from dIVA_L import dIVA_L


def test1 ( X, W_init, A, k, i ) :
    '''
    Fix number of subjects per site, increase sites.
    
    Returns graph of Joint ISI values
    '''
    
    N,R,K = X.shape
    list_of_sites = []
    master = dIVA_L()
    
    file = open("test1_%i_%i" % (k, i), "w")
    
    ## Doing two subjects per site, seems easiest
    for kk in range(K/2) :
        W = []
        X_m.append(X[:,:,kk:kk+2])
        W_m.append(W_init[:,:,kk:kk+2])
        dIVA_L(X, W_init)
        for site in list_of_sites :
            W.append(site.W[:,:,0])
            W.append(site.W[:,:,1])
        
        values.append( (kk, joint_ISI(W,A)) ) 
        file.write( "%i, %i, \n" % values[-1])
    
    ISI_values = [ value_pair[1] for value_pair in values ]
    X_values   = [ value_pair[0] for value_pair in values ]
    
    plt.plot(X_values, ISI_values)
    plt.title("test1 %i %i" % (k, i))
    plt.xlabel("Number of Sites")
    plt.ylabel("ISI Value")
    plt.savefig("test1_%i_%i_figure" % (k, i))
    plt.close('all')



def test2 ( X, W_init, A, k, i ) :
    '''
    Fix number of sites, increase number of subjects
        per site
    
    Returns graph of Joint ISI values
    '''
    N,R,K = X.shape
    master = dIVA_L()
    
    file = open("test2_%i_%i" % (k, i))
    
    ## Doing 5 total sites, seems easiest
    for kk in range(K/5) :
        list_of_sites = []
        
        count = 0
        for KK in range(5) :
            list_of_sites.append( local(X[:,:,count:count+KK+1], W_init=W_init[:,:,count:count+KK+1]) )
            count += KK + 1
        
        master.fit( list_of_sites )
        W = []
        
        for site in list_of_sites :
            for KK in range(kk) :
                W.append(site.W[:,:,KK])
        
        values.append( (kk, joint_ISI(W,A)) )
        file.write( "%i, %i, \n" % values[-1])    
    
    ISI_values = [ value_pair[1] for value_pair in values ]
    X_values   = [ value_pair[0] for value_pair in values ]
    
    plt.plot(X_values, ISI_values)
    plt.title("test1 %i %i" % (k, i))
    plt.xlabel("Number of Sites")
    plt.ylabel("ISI Value")
    plt.savefig("test2_%i_%i_figure" % (k, i))
    plt.close('all')



def test3 ( X, W_init, A, k, i ) :
    '''
    Fix total number of subjects, increase number of sites, 
        decrease total number of subjects per site
    
    Returns graph of Joint ISI values
    '''
    N,R,K = X.shape
    master = dIVA_L()
    
    file = open("test3_%i_%i" % (k, i))
    
    ## 
    for kk in range( 1, K+1 ) :
        subj_per_site = K/kk
        list_of_sites = []
        
        count = 0
        for KK in range(kk) :
            list_of_sites.append( local(X[:,:,count:count+subj_per_site], W_init[:,:,count:count+subj_per_site]) )
            count += subj_per_site
        
        master.fit(list_of_sites)
        W = []
        
        for site in list_of_sites :
            for KK in range(kk) :
                W.append( site.W[:,:,KK] )
        
        values.append( (kk, joint_ISI(W,A)) )
        file.write( "%i, %i, \n" % values[-1])    
        
    ISI_values = [ value_pair[1] for value_pair in values ]
    X_values   = [ value_pair[0] for value_pair in values ]
    
    plt.plot(X_values, ISI_values)
    plt.title("test1 %i %i" % (k, i))
    plt.xlabel("Number of Sites")
    plt.ylabel("ISI Value")
    plt.savefig("test2_%i_%i_figure" % (k, i))
    plt.close('all')
