import numpy as np
from cross_disper_mat import cross_disper_mat



def compute_Master_disper (list_of_disper) :
    N = list_of_disper[0].shape[2]
    num_subjects = 0
    for disper in list_of_disper :
        num_subjects += disper.shape[0]
    m_disper = np.zeros((num_subjects, num_subjects, N))
    
    for comp_num in range(N) :
        count = 0
        ## What component are we doing?
        for k in range(len(list_of_disper)) :
            kk_count = 0
            ## What column do we want
            for kk in range(k+1) :
                ## What row do we want
                if kk == 0 :
                    num_sub = list_of_disper[k].shape[0]
                    m_disper[count:count+num_sub, 
                             count:count+num_sub, 
                             comp_num] = list_of_disper[k][:,:,comp_num]
                else :
                    num_subc = list_of_disper[k ].shape[1]
                    num_subr = list_of_disper[kk-1].shape[0]
                    value = cross_disper_mat(list_of_disper[kk-1][:,:,comp_num], 
                                             list_of_disper[k] [:,:,comp_num])
                    m_disper[count-kk_count:count-kk_count+num_subr, 
                             count:count+num_subc, 
                             comp_num] = value
                    
                    m_disper[count:count+num_subc, 
                             count-kk_count:count-kk_count+num_subr, 
                             comp_num] = value.T
                    
                kk_count += list_of_disper[kk].shape[0]
            count += list_of_disper[k].shape[0]
    return m_disper