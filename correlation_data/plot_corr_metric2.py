import matplotlib.pyplot as plt
from numpy import abs, sum
from sys import argv
from multiprocessing import Pool

from make_corr_heatmap import get_sum_squared, make_corr_mat

def main ( name, list_subj) :
    comp = range(20)
    algorithm = partial_algorithm(list_subj)
    
    pool = Pool(20)
    
    pool.map(algorithm, comp)

def partial_algorithm (list_subj) :
    def algorithm (comp) :
        data = []
        for i in range(1, 11) :
            corr = get_sum_squared(n, list_subj[0:2**i])
            corr = make_corr_mat(corr)
            print i, l1_norm(corr)
            data.append( (i, l1_norm(corr)) )
            
        fil = open("correlation_metric_" + "comp" + index2(n) + ".txt", "w")
        
        for i in range(10) :
            fil.write(str(data[i][1]) + "\n")
        fil.close()
        
        plt.semilogx()
        plt.semilogy()
        plt.plot(data)
        plt.show()
        plt.savefig("correlation_metric_comp" + index2(n) + "_subj" + index4(2**i) + ".png")
    
    return algorithm

def l1_norm (A) :
    N, N = A.shape
    return sum(abs(A)) / (N**2)

def index4 (n) :
    if n < 10 : return "000" + str(n)
    if n < 100 : return "00" + str(n)
    if n < 1000 : return "0" + str(n)
    else : return str(n)

def index2 (n) :
    if n < 10 : return "0" + str(n)
    else : return str(n)

if __name__ == "__main__" :
    main(argv
