import matplotlib.pyplot as plt
from numpy import log10
from sys import argv

def main(list_subj) :
    data = []
    for i in list_subj :
        tmp_fil = open(list_subj[i], "r+")
        tmp_ls  = []
        text = tmp_fil.readline()
        while text != '' :
            tmp_ls.append(log10(float(text)))
            text = tmp_fil.readline()
        tmp_fil.close()
        data.append(tmp_ls)
    
    plt.figure()
    plt.boxplot(data)

if __name__=="__main__" :
    argv.pop(0)
    main(argv)
