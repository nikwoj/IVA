import scipy.optimize as opt
import numpy as np

def f(X) :
    
    y = 0
    for x in range(3) :
        for k in range(len(X[x])) :
            y += k**x
    
    return y
    
def g(X) :
    
    y = 0
    for x in range(len(X)) :
        for k in range(len(X[x])) :
            y += x - k
    return y

def h(X) :
    y = 0
    for x in X :
        y += np.abs(np.sum(np.sum(np.sum(X))))
    
    return y

def k(X) :
    y=0 
    for x in X :
        for k in x :
            y += abs(k)
    return y