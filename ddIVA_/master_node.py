import numpy as np
from numpy import sqrt, sum, zeros, dot
from munkres import Munkres

class master_node() :
    '''
    N = number of components
    R = number of samples
    KK = The total number of subjects involved in algorithm
    '''
    def initiate(self, KK) :
        self.KK = sum(KK)
    
    def order_comp(self, list_subj) :
        orderings = optimize_corr(list_subj)
        return orderings
    
    def master_step(self, YtY, w_value, cost, it, verbose) :
        '''
        Take in current values for YtY, log(abs(det(W))) and output
            cost, whether or not to terminate
        '''
        sqrtYtY = sqrt(YtY)
        sqrtYtYInv = 1 / sqrtYtY
        N, T = sqrtYtY.shape
       
        cost[it] = master_cost(w_value, sqrtYtY, self.KK)
        term = terminate(cost, it, verbose)
        backtrack = backtracking(cost, it)
        
        return sqrtYtYInv, backtrack, cost, term

def optimize_corr(list_subj) : 
    N,_ = list_subj[0].shape
    orderings = [[(i,i) for i in range(N)]]
    P = len(list_subj)
    m = Munkres()
    for p in range(1, P) :
        covar = dot(list_subj[0], list_subj[p].T)
        orderings.append( m.compute(covar) )
    
    return orderings

def master_cost(w_value, sqrtYtY, KK):
    N,R  = sqrtYtY.shape
    cost = sum(sqrtYtY) / R
    cost -= w_value
    cost /= (N*KK)
    return cost

def terminate(cost, it, verbose) :
    if it == 1 :
        return 1
    else :
        term = abs(cost[it-1] - cost[it]) / abs(cost[it])
        if verbose :
            print "Step %d \t Cost %f \t W Change %f" % (it, cost[it], term)
        return term

def backtracking(cost, it) :
    if it > 1 :
       if cost[it] < min(cost[0:it]) :
            return False
       else :
            return True
    else :
       return False
