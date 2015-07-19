import numpy as np



def k(W) :
    np.random.seed(0)
    W = np.dot(W, np.random.rand(W.shape[0], W.shape[0]))
    W = W*W
    return np.sum(W)