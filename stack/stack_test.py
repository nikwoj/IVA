import numpy as np
from stack import *

def main() :
    A = np.random.rand(100,100)
    a = stack(A)
    print np.all(A == unstack(a))  
    

if __name__ == "__main__" :
    main()