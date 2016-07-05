from ica import pca_whiten
from numpy import dot
from scipy.io import loadmat, savemat
from sys import argv

## NOTE : To use, call from command line as 
##
##  python pca_preprocess name_save_as components_use files_subjects files_matrices
##
## Should be same number of files_subjects as files_matrices
## Should correspond to one another
##

def index(k) :
    if k < 10 :
        return "000%d" % k
    if k < 100 :
        return "00%d" % k
    if k < 1000 :
        return "0%d" % k
    else :
        return "%d" % k

def pca_preprocess(name, comp, subjects, matrices, verbose=True) :
    assert len(subjects) == len(matrices)
    K = len(subjects)
    for k in range(K) :
        matrix  = loadmat(matrices[k])['A']
        subject = dot(matrix, loadmat(subjects[k])['S'])
        X_white, wht, de_wht = pca_whiten(subject, comp)
        savemat(name + index(k+1) + ".mat", {'X_white':X_white,'wht':wht,'de_wht':de_wht})
        if verbose :
            print "subject " + str(k+1) + " done"

if __name__=="__main__" :
    argv.pop(0)
    name = argv.pop(0)
    comp = int(argv.pop(0))
    n = len(argv)/2
    subjects = argv[:n]
    matrices = argv[n:]
    pca_preprocess(name, comp, subjects, matrices)
