import numpy as np
from numpy import sqrt, sum

def master_cost(w_value, sqrtYtY, KK):
    N,T = sqrtYtY.shape
    cost = sum(sqrtYtY) / T
    cost -= w_value
    cost /= (N*sum(KK))
    return cost

def terminate(cost, it, verbose) :
    if it == 1 :
        return 1
    else :
        term = abs(cost[it-1] - cost[it]) / abs(cost[it])
        if verbose :
            print "Step %d \t Cost %f \t W Change %f" % (it, cost[it], term)
        return term

class master_node() :
    '''
    N = number of components
    T = number of samples
    KK = a list of number of subjects at every local_node
    '''
    def initiate(self, YtY, KK) :
        self.KK = KK
        sqrtYtYInv = 1 / sqrt(YtY)
        return sqrtYtYInv
    
    def master_step(self, YtY, w_value, cost, it, verbose) :
        '''
        Take in current values for YtY, log(abs(det(W))) and output
            cost, whether or not to terminate
        '''
        # print sum(YtY, 0)
        sqrtYtY = sqrt(YtY)
        # print sqrtYtY
        sqrtYtYInv = 1 / sqrtYtY
        N, T = sqrtYtY.shape
        
        cost[it] = master_cost(w_value, sqrtYtY, self.KK)
        term = terminate(cost, it, verbose)
        return sqrtYtYInv, cost, term
