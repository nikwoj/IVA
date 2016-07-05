from scipy.stats import wilcoxon
from numpy import mean, log10
from sys import argv


## Takes in two sets of files, compiles each set into an 
## array, log10's the arrays,  and says whether or not 
## mean of one is statistically different from mean of 
## other.
##
## Example call use :
## 
##     python test_sig.py fil1 fil2
##

def test_sig(fil1, fil2) :
    l1 = []
    l2 = []
    print fil1
    print fil2
    fil1 = open(fil1, "r+")
    fil2 = open(fil2, "r+")
    
    ## Don't read the \n character
    text = fil1.readline()[:-1]
    while text != "" :
        l1.append(log10(float(text)))
        text = fil1.readline()[:-1]
    
    text = fil2.readline()[:-1]
    while text != "" :
        l2.append(log10(float(text)))
        text = fil2.readline()[:-1]
    
    stats = wilcoxon(l1, l2)
    
    print "[PCA IVAG IVAL] ISI  : ", mean(l1)
    print "[PCA      IVAL] ISI  : ", mean(l2)
    print stats
    return stats

if __name__ == "__main__" :
    fil1 = argv.pop(1)
    fil2 = argv.pop(1)
    test_sig(fil1, fil2)
