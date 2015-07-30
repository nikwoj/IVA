import autograd.numpy as np

from autograd import value_and_grad

from gradient import set_functions
from auto_grad import set_para

def main() :
    X = np.random.rand(10,30,4)
    
    compute_cost, W = set_para(X)
    auto_gradient = value_and_grad(compute_cost)
    
    cost_and_grad, W = set_functions(X)
    
    for i in range(10) :
        W = np.random.rand(10,10,4)
        a = auto_gradient(W)
        b = cost_and_grad(W)
        if np.max(a[1]-b[1]) > .01 :
            print "Why"
        
        print a[0] - b[0]
            
            
if __name__ == "__main__" : 
    main()