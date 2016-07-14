import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from sys import argv

## NOTE : Given spatial map info, a component number a spatial map and a name,
## ## outputs and saves png file with the spatial map image, with name

## ipython spatial_map_info.mat 2 SCV_IVA_caseNik_r001_subj0001.mat spatial_map_comp2_subj1.png

def make_spatial_map(name, msk, comp, mat) :
    ## Dimension of image
    R = 206
    
    smap = np.zeros((R**2))
    
    i = 0
    for r in range(R**2) :
        if msk[0][r]==1 :
            smap[r] = mat[comp,i]
            i += 1
        else :
            smap[r] = 1
    
    smap = smap.reshape(R,R)
    fig = plt.imshow(smap, cmap=plt.cm.hot)
    plt.axis('off')
    plt.colorbar()
    plt.show()
    plt.savefig(name, bbox_inches='tight')
    #plt.savefig(name)

if __name__=="__main__" :
    name = argv.pop(1)
    msk  = loadmat(argv.pop(1))['msk']
    comp = argv.pop(1)
    mat  = loadmat(argv.pop(1))['S']
    make_spatial_map(name, msk, comp, mat)
