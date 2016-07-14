import matplotlib.pyplot as plt

from sys import argv

def main(name, ylabel, xlabel, list_files) :
    data = []
    i = 2
    for fil in list_files :
        data.append( get_data(fil) )
        i += 1
    a, axes = plt.subplots(nrows=1,ncols=1,figsize=(6,6))
    axes.semilogy()
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    axes.boxplot(data)
    #axes.set_xlim([1,len(list_files)+1])
    plt.show()
    a.savefig(name)

def get_data(fil) :
    data = []
    fil = open(fil, "r+")
    text = fil.readline()
    while text != "" :
        data.append(float(text))
        text = fil.readline()
    
    return data

if __name__=="__main__" :
    argv.pop(0)
    name = argv.pop(0)
    ylabel = argv.pop(0)
    xlabel = argv.pop(0)
    main(name, ylabel, xlabel, argv)
