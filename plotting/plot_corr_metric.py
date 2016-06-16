import numpy as np
import matplotlib.pyplot as plt

from numpy import sum, abs

from make_corr_heatmap import get_sum_squared_mat, get_corr_comp_mat
from sys import argv
def make_metric_plot(comp, subjects) :
    B = get_sum_squared_mat(comp, subjects)
    B = get_corr_comp_mat(B)
    
    x2,   y2   = subj_site2(B)
    x10,  y10  = subj_site10(B)
    x100, y100 = subj_site100(B)
    plt.semilogx()
    plt.plot(x2,   y2)
    plt.plot(x10,  y10)
    plt.plot(x100, y100)
    plt.show()
    plt.savefig("correlation_" + str(comp) + "_metric.png")   

def l1_norm(A) :
    ## HOW THE FUCK DOES NUMPY NOT HAVE THIS!
    ## ||A||_1 = \sum_{i,j} |a_{i,j}|
    return sum(abs(A))

def l1_metric(A) : 
    ## Provides metric for how correlated a correlation
    ## matrix is. Returns number 0<num<1
    N,_ = A.shape
    return l1_norm(A) / N**2

def subj_site2(B) :
    y2 = []
    x2 = []
    for i in range(1, 10) :
        y2.append(l1_metric(B[0:2**i, 0:2**i]))
        x2.append(2**i)
    return x2, y2

def subj_site10(B) :
    y10 = []
    x10 = []
    for i in range(1, 7) :
        y10.append(l1_metric(B[0:10**i, 0:10**i]))
        x10.append(10**i)
    return x10, y10

def subj_site100(B) :
    y100 = []
    x100 = []
    for i in range(1, 4) :
        y100.append(l1_metric(B[0:100**i, 0:100**i]))
        x100.append(100**i)
    return x100, y100

if __name__=="__main__" :
    argv.pop(0)
    comp = int(argv.pop(0))
    make_metric_plot(comp, argv)
