import matplotlib.pyplot as plt
from numpy import abs, sum
from numpy.linalg import det
from sys import argv
from multiprocessing import Pool
from scipy.io import savemat


from make_corr_heatmap import get_sum_squared_mat, get_corr_comp_mat

def main () :
    comp = range(20)
    
    pool = Pool(20)
    
    pool.map(algorithm, comp)

def algorithm (comp) :
    data = []
    for i in range(1, 8) :
        corr = get_sum_squared_mat(comp, 2**i)
        corr = get_corr_comp_mat(corr)
        savemat("correlation_mat_comp%s_subj%s.mat"%(index2(comp), index4(2**i)), {'corr':corr})
        #metric_val = l1_metric(corr)
        metric_val = det(corr)
        print "num_subj: %d comp: %d metric_val: %d"%(i, metric_val)
        data.append( (i, metric_val) )
        
    fil = open("correlation_metric_comp" + index2(n) + ".png")
    
    for i in range(10) :
        fil.write(str(data[i][1]) + "\n")
    fil.close()
    
    plt.xlabel("Log2(Subjects)")
    plt.ylabel("Log(ISI)")
    plt.semilogx()
    plt.semilogy()
    plt.plot(data)
    plt.show()
    plt.savefig("correlation_metric_comp" + index2(n) + ".png")

def l1_metric (A) :
    N, N = A.shape
    return sum(abs(A)) / (N*N)

def index4 (n) :
    if n < 10 : return "000" + str(n)
    if n < 100 : return "00" + str(n)
    if n < 1000 : return "0" + str(n)
    else : return str(n)

def index2 (n) :
    if n < 10 : return "0" + str(n)
    else : return str(n)

if __name__ == "__main__" :
    main()
